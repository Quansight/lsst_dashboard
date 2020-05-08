import yaml
from functools import partial
from pathlib import Path

import dask.dataframe as dd
import os

import numpy as np
import pandas as pd

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

    def __init__(self, path, coadd_version="unforced"):
        self.path = Path(path)
        self.coadd = {}
        self.visits = None
        self.visits_by_metric = {}
        self.metrics = []
        self.failures = {}  # this functionality no longer works
        self.flags = []
        self.filters = []
        self.tracts = []
        self.stats = {}
        self.coadd_version = coadd_version

    def connect(self):
        print("-- read coadd/visits summary stats tables and generate metadata")
        self.read_summary_stats()
        # use coadd table to populate filters & tracts
        coadd_version = self.coadd_version
        self.filters = list(self.stats[f"coadd_{coadd_version}"].index.unique(level=0))
        self.tracts = list(self.stats[f"coadd_{coadd_version}"].index.unique(level=1))

        print(f"-- read {coadd_version} coadd table --")
        self.fetch_coadd_table(coadd_version=coadd_version)

        print("-- generate other metadata fields --")
        self.post_process_metadata()

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

        print(f"...loading dataset ({filter_name}, {metrics})...")
        coadd_df = (
            read_dataset_as_ddf(**karto_kwargs).dropna(how="any")
            # .set_index('filter')
            .compute()
        )
        print("loaded.")

        # coadd_df = dd.from_pandas(coadd_df, chunksize=100000)

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

    def read_summary_stats(self):
        for table in ["CoaddTable_unforced", "CoaddTable_forced", "VisitTable"]:
            name = table.replace("Table", "").lower()
            path = self.path.joinpath(f"analysis{table}_stats.parq")
            self.stats[name] = pd.read_parquet(path)

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
        self.metrics = (
            set(df.columns.to_list())
            - set(self.flags)
            - set(["patch", "dec", "psfMag", "ra", "filter", "dataset", "tract"])
        )

    def get_visits_by_metric_filter(self, filt, metric):

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
