import distributed
from dask import delayed
from pandas.testing import assert_series_equal
from kartothek.io.dask.dataframe import update_dataset_from_ddf, read_dataset_as_ddf
from kartothek.io.eager import read_dataset_as_dataframes
from storefact import get_store_from_url
from functools import partial
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import dask.dataframe as dd
import dask.array as da
from tqdm import tqdm

from lsst.daf.persistence import Butler

NULL_VALUE = {
    np.dtype("bool"): True,
    np.dtype("int32"): 0,
    np.dtype("int64"): 0,
    np.dtype("float32"): np.nan,
    np.dtype("float64"): np.nan,
    np.dtype("O"): np.nan,
}

DTYPE_MAP = {
    "DOUBLE": "float64",
    "FLOAT": "float32",
    "INT64": "int64",
    "BOOLEAN": "bool",
    "INT32": "int32",
    "BYTE_ARRAY": "object",
}


def get_dtypes(filename, dtype_map=DTYPE_MAP, columns=None, skip_columns=None):
    schema = pq.ParquetFile(filename).schema
    if columns is None:
        columns = schema.names
    if skip_columns is None:
        skip_columns = ["id"]
    dtypes = pd.Series({x.name: dtype_map[x.physical_type] for x in schema if x.name != "id"})
    dtypes.index.name = "column"
    return dtypes.sort_index()


def dtypes_identical(dtypes1, dtypes2, debug=False):
    try:
        assert_series_equal(dtypes1, dtypes2)
        return True
    except AssertionError:
        if debug:
            bad = dtypes1 != dtypes2
            print(dtypes1[bad], len(dtypes1))
            print(dtypes2[bad], len(dtypes2))
            raise
        return False


def count_schemas(filenames, **kwargs):
    """Returns list of pandas dtypes series, and corresponding list of occurrences
    """
    schemas = []
    fnames = []
    counts = []
    for f in tqdm(filenames, desc="Reading schemas"):
        #         print(f)
        dtypes = get_dtypes(f, **kwargs)
        if not schemas:
            schemas.append(dtypes)
            fnames.append(f)
            counts.append(1)
        else:
            found = False
            for i, dt in enumerate(schemas):
                if dtypes_identical(dtypes, dt):
                    counts[i] += 1
                    found = True
                    break
            if not found:
                schemas.append(dtypes)
                fnames.append(f)
                counts.append(1)

    return schemas, fnames, counts


def fix_dtypes(df, target_dtypes, test_col="coord_ra"):
    if all(df.dtypes == target_dtypes):
        return df
    else:
        cols_to_fix = []
        df_new = df.dropna(subset=[test_col])  # get rid if necessary col is nan
        for c in df_new.columns:
            if df_new[c].dtype == target_dtypes[c]:
                continue
            else:
                target_dtype = target_dtypes[c]
                df_new.loc[df_new[c].isnull(), c] = NULL_VALUE[target_dtype]
                df_new[c] = df_new[c].astype(target_dtype)
        return df_new


def get_metrics():
    return [
        "base_Footprint_nPix",
        "Gaussian-PSF_magDiff_mmag",
        #             'CircAper12pix-PSF_magDiff_mmag',
        "Kron-PSF_magDiff_mmag",
        "CModel-PSF_magDiff_mmag",
        "traceSdss_pixel",
        "traceSdss_fwhm_pixel",
        "psfTraceSdssDiff_percent",
        "e1ResidsSdss_milli",
        "e2ResidsSdss_milli",
        "deconvMoments",
        "compareUnforced_Gaussian_magDiff_mmag",
        #             'compareUnforced_CircAper12pix_magDiff_mmag',
        "compareUnforced_Kron_magDiff_mmag",
        "compareUnforced_CModel_magDiff_mmag",
        "base_PsfFlux_instFlux",
        "base_PsfFlux_instFluxErr",
    ]


def get_flags():
    return [
        "calib_psf_used",
        "calib_psf_candidate",
        "calib_photometry_reserved",
        "merge_measurement_i2",
        "merge_measurement_i",
        "merge_measurement_r2",
        "merge_measurement_r",
        "merge_measurement_z",
        "merge_measurement_y",
        "merge_measurement_g",
        "merge_measurement_N921",
        "merge_measurement_N816",
        "merge_measurement_N1010",
        "merge_measurement_N387",
        "merge_measurement_N515",
        "qaBad_flag",
    ]


class DatasetPartitioner(object):
    """Partitions datasets with ['filter', 'tract'] dataId keys
    """

    _default_dataset = None
    _default_df_chunk_size = 20

    _default_partition_on = ("filter", "tract")

    def __init__(
        self,
        butlerpath,
        destination=None,
        dataset=None,
        partition_on=None,
        bucket_by="psfMag",
        categories=None,
        engine="pyarrow",
        sample_frac=None,
        num_buckets=8,
        df_chunk_size=None,
    ):

        if isinstance(butlerpath, Butler):
            self._butler = butlerpath
        else:
            self._butler = Butler(butlerpath)

        if dataset is None:
            dataset = self._default_dataset

        self.dataset = dataset

        if partition_on is None:
            partition_on = self._default_partition_on
        self.partition_on = partition_on

        self.bucket_by = bucket_by
        self.categories = categories

        if destination is None:
            destination = f"{butlerpath}/ktk"
        self.destination = destination
        self.sample_frac = sample_frac
        self.num_buckets = num_buckets
        if df_chunk_size is None:
            df_chunk_size = self._default_df_chunk_size
        self.df_chunk_size = df_chunk_size

        self.stats_path = f"{self.destination}/{self.dataset}_stats.parq"

        self._store = None
        self.engine = engine
        self.metadata = self.butler.get("qaDashboard_info")
        self.dataIds = self.metadata[self.dataset]

        # self.filters = [filt for filt in self.metadata["visits"].keys()]
        # self.dataIds_by_filter = {
        #     filt: [d for d in self.dataIds if d["filter"] == filt] for filt in self.filters
        # }

        self._filenames = None
        # self._filenames_by_filter = None

    def __getstate__(self):
        d = self.__dict__
        d["_butler"] = None
        d["_store"] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d

    @property
    def store(self):
        if self._store is None:
            self._store = partial(get_store_from_url, "hfs://" + self.destination)
        return self._store

    @property
    def butler(self):
        if self._butler is None:
            self._butler = Butler(butlerpath)
        return self._butler

    @property
    def filenames(self):
        if self._filenames is None:
            filenames = []
            for dataId in tqdm(self.dataIds, desc=f"Getting filenames for {self.dataset} from Butler"):
                filename = self.butler.get(self.dataset, **dataId).filename
                filenames.append(filename)
            self._filenames = filenames
        return self._filenames

    def normalize_df(self, df):
        """
        append ra, dec, psfMag to dataframe and cleanup
        """

        # TODO I think the partitioning destroys the original indexing if the index numbers are important we may need to do a reset_index()

        if "base_PsfFlux_instFlux" in df.columns:
            flux_col = "base_PsfFlux_instFlux"
        elif "PsFlux" in df.columns:
            flux_col = "PsFlux"
        elif "iPsFlux" in df.columns:
            flux_col = "iPsFlux"
        else:
            raise ValueError("No PSF Flux column name?")

        df = (
            df.assign(
                ra=da.rad2deg(df.coord_ra),
                dec=da.rad2deg(df.coord_dec),
                psfMag=-2.5 * da.log10(df[flux_col]),
            )
            .replace(np.inf, np.nan)
            .replace(-np.inf, np.nan)
            .rename(columns={"patchId": "patch", "ccdId": "ccd"})
        )

        if self.categories:
            df = df.categorize(columns=self.categories)

        # Drop duplicate column names
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def get_metric_columns(self):
        return get_metrics()

    def get_flag_columns(self):
        return get_flags()

    def get_columns(self):
        return None
        # return list(
        #     set(self.get_metric_columns() + self.get_flag_columns() + ["coord_ra", "coord_dec", "patchId"])
        # )

    def infer_target_dtypes(self, filenames, dataId, columns=None):

        # Find file with the most common schema
        schemas, fnames, counts = count_schemas(filenames, columns=columns)
        filename = fnames[np.argmax(counts)]

        df = (
            pd.read_parquet(filename, columns=columns, engine=self.engine)
            .assign(**dataId)
            .sort_index(axis=1)
        )

        return df.dtypes

    def df_generator(self, dataIds, filenames, columns, msg=None):
        if msg is None:
            desc = f"Building dask dataframe for {self.dataset}"
        else:
            desc = f"Building dask dataframe for {self.dataset} ({msg})"

        target_dtypes = self.infer_target_dtypes(filenames, dataIds[0], columns=columns)
        for filename, dataId in tqdm(zip(filenames, dataIds), desc=desc, total=len(dataIds),):
            df = delayed(pd.read_parquet(filename, columns=columns, engine=self.engine))
            df = delayed(pd.DataFrame.assign)(df, **dataId)
            df = delayed(pd.DataFrame.sort_index)(df, axis=1)
            df = delayed(fix_dtypes)(df, target_dtypes)
            yield df

    def get_df(self, dataIds, filenames, msg=None):
        columns = self.get_columns()

        if len(dataIds) > 0:
            df = dd.from_delayed(self.df_generator(dataIds, filenames, columns, msg=msg))

            if self.sample_frac:
                df = df.sample(frac=self.sample_frac)

            df = self.normalize_df(df)

            return df
        else:
            return None

    def iter_df_chunks(self, chunk_dfs=False):
        dataIds = self.dataIds
        filenames = self.filenames

        if chunk_dfs:
            n_chunks = len(dataIds) // self.df_chunk_size or 1
        else:
            n_chunks = 1
        for i in range(n_chunks):
            msg = f"{i + 1} of {n_chunks}"
            yield self.get_df(dataIds[i::n_chunks], filenames[i::n_chunks], msg=msg)

    @property
    def ktk_kwargs(self):
        return dict(
            dataset_uuid=self.dataset,
            store=self.store,
            table="table",
            shuffle=True,
            num_buckets=self.num_buckets,
            bucket_by=self.bucket_by,
            partition_on=self.partition_on,
        )

    def partition(self, chunk_dfs=False):
        """Write partitioned dataset using kartothek
        """
        if len(self.filenames) == 0:
            return

        for i, df in enumerate(self.iter_df_chunks(chunk_dfs=chunk_dfs)):
            if df is not None:
                n_cols = len(df.columns)
                n_rows = len(df)
                print(
                    f"... ...ktk repartitioning {self.dataset} (chunk {i + 1}; {n_cols} colums, {n_rows} rows)"
                )
                graph = update_dataset_from_ddf(df, **self.ktk_kwargs)
                graph.compute()

    def load_from_ktk(self, predicates, columns=None, dask=True):
        ktk_kwargs = dict(
            dataset_uuid=self.dataset, predicates=predicates, store=self.store, columns={"table": columns},
        )
        #         print(ktk_kwargs)
        if dask:
            ktk_kwargs["table"] = "table"
            return read_dataset_as_ddf(**ktk_kwargs)
        else:
            ktk_kwargs["tables"] = ["table"]
            datalist = read_dataset_as_dataframes(**ktk_kwargs)
            if datalist:
                return read_dataset_as_dataframes(**ktk_kwargs)[0]["table"]
            else:
                raise IOError(f"No data returned for {ktk_kwargs}.")

    def load_dataId(self, dataId, columns=None, dask=False, raise_exception=False):
        predicates = [[(k, "==", v) for k, v in dataId.items()]]
        try:
            return self.load_from_ktk(predicates, columns=columns, dask=dask)
        except IOError:
            if raise_exception:
                raise
            else:
                print(f"No {self.dataset} data available for {dataId}, columns={columns}")
                return pd.DataFrame()

    def describe_dataId(self, dataId, dask=False, **kwargs):
        df = self.load_dataId(dataId, dask=dask)
        return df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna(how="any").describe(**kwargs)

    def get_stats_list(self, dataIds=None):
        if dataIds is None:
            dataIds = self.dataIds

        fn = partial(describe_dataId, store=self.store, dataset=self.dataset)

        client = distributed.client.default_client()

        futures = client.map(fn, dataIds)
        results = client.gather(futures)
        return results

    def compute_stats(self, dataIds=None):
        if dataIds is None:
            dataIds = self.dataIds

        stats_list = self.get_stats_list(dataIds)

        dfs = []
        for dataId, stats in zip(dataIds, stats_list):
            index = pd.MultiIndex.from_tuples(
                [(*dataId.values(), s) for s in stats.index], names=[*dataId.keys(), "statistic"]
            )
            columns = stats.columns
            df = pd.DataFrame(stats.values, index=index, columns=columns)
            dfs.append(df)

        return pd.concat(dfs, sort=True)

    def write_stats(self, dataIds=None):
        if len(self.filenames) == 0:
            return

        stats = self.compute_stats(dataIds=dataIds)
        stats.to_parquet(self.stats_path)

    def load_stats(self, columns=None):
        if not os.path.exists(self.stats_path):
            self.write_stats()

        return pd.read_parquet(self.stats_path, columns=columns)


class PipeAnalysisPartitioner(DatasetPartitioner):
    def get_columns(self):
        return list(
            set(self.get_metric_columns() + self.get_flag_columns() + ["coord_ra", "coord_dec", "patchId"])
        )


class CoaddForcedPartitioner(PipeAnalysisPartitioner):
    _default_dataset = "analysisCoaddTable_forced"
    _default_partition_on = ("filter", "tract")


class CoaddUnforcedPartitioner(PipeAnalysisPartitioner):
    _default_dataset = "analysisCoaddTable_unforced"

    def get_metric_columns(self):
        return list(
            set(get_metrics())
            - {
                "compareUnforced_CModel_magDiff_mmag",
                "compareUnforced_Gaussian_magDiff_mmag",
                "compareUnforced_Kron_magDiff_mmag",
            }
        )


class VisitAnalysisPartitioner(PipeAnalysisPartitioner):
    _default_partition_on = ("filter", "tract", "visit")
    _default_dataset = "analysisVisitTable"
    # _default_df_chunk_size = 80

    def get_metric_columns(self):
        return list(
            set(get_metrics())
            - {
                "CModel-PSF_magDiff_mmag",
                "compareUnforced_CModel_magDiff_mmag",
                "compareUnforced_Gaussian_magDiff_mmag",
                "compareUnforced_Kron_magDiff_mmag",
            }
        )

    def get_columns(self):
        return super().get_columns() + ["ccdId", "filter", "tract", "visit"]


def describe_dataId(
    dataId, store, dataset, columns=None, percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
):
    """This wasn't working as a method with client.map because of deserialization problem
    """
    predicates = [[(k, "==", v) for k, v in dataId.items()]]
    ktk_kwargs = dict(
        dataset_uuid=dataset,
        predicates=predicates,
        store=store,
        tables=["table"],
        columns={"table": columns},
    )

    df = read_dataset_as_dataframes(**ktk_kwargs)[0]["table"]
    return (
        df.replace(np.inf, np.nan)
        .replace(-np.inf, np.nan)
        .dropna(how="any")
        .describe(percentiles=percentiles)
    )
