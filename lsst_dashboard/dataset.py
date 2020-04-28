import yaml
from functools import partial
from pathlib import Path

import dask.dataframe as dd
import os

import numpy as np

from kartothek.io.dask.dataframe import read_dataset_as_ddf
from storefact import get_store_from_url


METADATA_FILENAME = "dashboard_metadata.yaml"


class Dataset:
    """
    USAGE:
        d = Dataset(path)
        d.connect()
        d.init_data()
    """
    def __init__(self, path, tracts=None, filters=None):
        self.conn = None
        self.path = Path(path)
        self.coadd = {}
        self.visits = None
        self.visits_by_metric = {}
        self.metadata = {}
        self.metrics = []
        self.failures = {}
        self.flags = []
        self.tracts = tracts
        self.filters = filters if filters is not None else []

    def connect(self):

        self.parse_metadata_from_file()

        print("-- read coadd table --")
        self.fetch_coadd_table()  # currently ignoring forced/unforced

        print("-- generate other metadata fields --")
        self.post_process_metadata()

        print("-- read visit data --")
        self.fetch_visits_by_metric()

        print("-- done with reads --")

    def get_coadd_ddf_by_filter_metric(
        self, filter_name, metrics, tracts, coadd_version="unforced", warnings=[]
    ):

        for t in tracts:
            if t not in self.tracts:
                msg = "Selected tract {} missing in data".format(t)
                print("WARNING: {}".format(msg))
                warnings.append(msg)

        # filter out any tracts not in data
        valid_tracts = list(set(self.tracts).intersection(set(tracts)))

        if not valid_tracts:
            msg = "No Valid tracts selected...using all tracts"
            print("WARNING: {}".format(msg))
            valid_tracts = self.tracts
            warnings.append(msg)

        predicates = [[("tract", "in", valid_tracts), ("filter", "==", filter_name)]]

        dataset = "analysisCoaddTable_{}".format(coadd_version)

        columns = metrics + self.flags + ["ra", "dec", "filter", "psfMag", "patch"]

        store = partial(get_store_from_url, "hfs://" + str(self.path))

        karto_kwargs = dict(
            predicates=predicates, dataset_uuid=dataset, columns=columns, store=store, table="table"
        )

        coadd_df = (read_dataset_as_ddf(**karto_kwargs)
                    .repartition(partition_size="4GB")
                    .replace(np.inf, np.nan)
                    .replace(-np.inf, np.nan)
                    .dropna(how='any')
                    .set_index('filter')
                    .persist())

        return coadd_df

    def get_patch_count(self, filters, tracts, coadd_version="unforced"):

        return 1

        predicates = []

        if filters:
            predicates.append(("filter", "in", filters))

        if tracts:
            predicates.append(("tract", "in", tracts))

        dataset = "analysisCoaddTable_{}".format(coadd_version)

        columns = ["patch"]

        store = partial(get_store_from_url, "hfs://" + str(self.path))

        if predicates:

            coadd_df = read_dataset_as_ddf(
                predicates=[predicates], dataset_uuid=dataset, columns=columns, store=store, table="table"
            )
        else:
            coadd_df = read_dataset_as_ddf(dataset_uuid=dataset, columns=columns, store=store, table="table")

        return coadd_df.drop_duplicates().count().compute()["patch"]

    def parse_metadata_from_file(self):
        if self.path.joinpath(METADATA_FILENAME).exists():
            self.metadata_path = self.path.joinpath(METADATA_FILENAME)
        else:
            self.metadata_path = Path(os.environ.get("LSST_META", os.curdir)).joinpath(
                self.path.name, METADATA_FILENAME
            )

        with self.metadata_path.open("r") as f:
            self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
            self.failures = self.metadata.get("failures", {})
            if self.tracts is None:
                self.tracts = list(set(x for v in self.metadata["visits"].values() for x in v.keys()))

    def fetch_coadd_table(self, coadd_version="unforced"):
        table = "qaDashboardCoaddTable"
        store = partial(get_store_from_url, "hfs://" + str(self.path))
        print(str(self.path))
        predicates = [[("tract", "in", self.tracts)]]
        dataset = "analysisCoaddTable_{}".format(coadd_version)

        coadd_df = read_dataset_as_ddf(
            predicates=predicates, dataset_uuid=dataset, store=store, table="table"
        )

        self.coadd[table] = coadd_df

    def post_process_metadata(self):
        df = self.coadd["qaDashboardCoaddTable"]
        self.flags = df.columns[df.dtypes == bool].to_list()
        self.filters = list(self.metadata["visits"].keys())
        self.metrics = (
            set(df.columns.to_list())
            - set(self.flags)
            - set(["patch", "dec", "psfMag", "ra", "filter", "dataset", "dir0", "tract"])
        )

    def fetch_visits(self):
        store = partial(get_store_from_url, "hfs://" + str(self.path))
        predicates = [[("tract", "in", self.tracts)]]
        self.visits = read_dataset_as_ddf(
            predicates=predicates, dataset_uuid="analysisVisitTable", store=store, table="table"
        )

    def get_visits_by_metric_filter(self, filt, metric):

        store = partial(get_store_from_url, 'hfs://' + str(self.path))

        columns = ['filter', 'tract', 'visit', 'calib_psf_used',
                   'calib_psf_candidate', 'calib_photometry_reserved',
                   'qaBad_flag', 'ra', 'dec', 'psfMag'] + [metric]

        visits_ddf = read_dataset_as_ddf(dataset_uuid="analysisVisitTable",
                                         predicates=[[('filter', '==', filt)]],
                                         store=store,
                                         columns=columns,
                                         table='table')
        store = partial(get_store_from_url, "hfs://" + str(self.path))

        columns = [
            "filter",
            "tract",
            "visit",
            "calib_psf_used",
            "calib_psf_candidate",
            "calib_photometry_reserved",
            "qaBad_flag",
            "ra",
            "dec",
            "psfMag",
        ] + [metric]

        visits_ddf = read_dataset_as_ddf(
            dataset_uuid="analysisVisitTable",
            predicates=[[("filter", "==", filt)]],
            store=store,
            columns=columns,
            table="table",
        )

        return visits_ddf[visits_ddf[metric].notnull()]

    def fetch_visits_by_metric(self):
        for filt in self.filters:
            self.visits_by_metric[filt] = {}
            for metric in self.metrics:
                try:
                    ddf = self.get_visits_by_metric_filter(filt, metric)
                except:
                    # raise
                    print("WARNING: problem loading visits for {} metric and {} filter".format(metric, filt))
                    ddf = None

                self.visits_by_metric[filt][metric] = ddf
