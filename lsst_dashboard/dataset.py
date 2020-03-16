import yaml
from functools import partial
from pathlib import Path

import dask.dataframe as dd
import os

from kartothek.io.dask.dataframe import read_dataset_as_ddf
from storefact import get_store_from_url

try:
    from lsst.daf.persistence import Butler
    from lsst.daf.persistence.butlerExceptions import NoResults
except:
    Butler = None

METADATA_FILENAME = 'dashboard_metadata.yaml'


class Dataset():
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

    def parse_metadata_from_file(self):
        if self.path.joinpath(METADATA_FILENAME).exists():
            self.metadata_path = self.path.joinpath(METADATA_FILENAME)
        else:
            self.metadata_path = Path(os.environ.get('LSST_META', os.curdir)).joinpath(self.path.name, METADATA_FILENAME)

        with self.metadata_path.open('r') as f:
            self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
            self.failures = self.metadata.get('failures', {})
            if self.tracts is None:
                self.tracts = list(set(x for v in self.metadata['visits'].values() for x in v.keys()))

    def parse_metadata_from_butler(self):
        try:
            print('-- connect to butler --')
            self.conn = Butler(str(self.path))
            self.metadata = self.conn.get('qaDashboard_metadata')
            self.failures = self.metadata.get('failures', {})
            if not self.filters:
                self.filters = list(self.metadata['visits'].keys())
            if not self.tracts:
                all_tracts = [list(self.metadata['visits'][filt].keys()) for filt in self.filters]
                self.tracts = list(set([int(y) for x in all_tracts for y in x]))
        except:
            raise
            print(f'{self.path} is not available in Butler attempting to read parquet files instead')

    def connect(self):

        print('-- read metadata file --')
        if Butler:
            self.parse_metadata_from_butler()
        else:
            self.parse_metadata_from_file()

        print('-- read coadd table --')
        self.fetch_coadd_table()  # currently ignoring forced/unforced

        # update metadata based on coadd table fields
        print('-- generate other metadata fields --')
        df = self.coadd['qaDashboardCoaddTable']
        self.flags = df.columns[df.dtypes == bool].to_list()

        if not Butler:
            self.filters = list(self.metadata['visits'].keys())

        self.metrics = set(df.columns.to_list()) - set(self.flags) - set(['patch', 'dec', 'label', 'psfMag',
                                                                         'ra', 'filter', 'dataset', 'dir0', 'tract'])
        print('-- read visit data --')
        self.fetch_visits_by_metric()
        print('-- done with reads --')

    def fetch_coadd_table(self, coadd_version='unforced'):

        table = 'qaDashboardCoaddTable'  # + coadd_version

        if Butler:
            filenames = [self.conn.get(table, tract=int(t)).filename for t in self.tracts]
        else:
            filenames = [str(self.path.joinpath(f'{table}-{t}.parq')) for t in self.tracts]

        column_map = {'patchId': 'patch', 'tractId': 'tract'}
        self.coadd[table] = dd.read_parquet(filenames).rename(columns=column_map).compute()


    def fetch_visits(self):

        table = 'qaDashboardVisitTable'

        if self.conn:
            pass
            # filenames = [self.conn.get(table, tract=int(t)).filename for t in self.tracts]
        else:
            filenames = [str(self.path.joinpath(f'{table}-{t}.parq')) for t in self.tracts]

        self.visits = dd.read_parquet(filenames, npartitions=16).rename(columns={'tractId': 'tract', 'visitId': 'visit', 'patchId': 'patch'})


    def fetch_visits_by_metric(self):
        for filt in self.filters:
            self.visits_by_metric[filt] = {}
            for metric in self.metrics:
                if self.conn:
                    filenames = []
                    for t in self.tracts:
                        try:
                            filenames.append(self.conn.get('qaDashboardVisitTable', tract=int(t), filter=filt, column=metric).filename)
                        except NoResults:
                            continue
                else:
                    filenames =  list(self.path.glob(f'./*{filt}*{metric}.parq'))

                column_map = {'tractId': 'tract', 'visitId': 'visit', 'patchId': 'patch'}
                self.visits_by_metric[filt][metric] = dd.read_parquet(filenames).rename(columns=column_map)



class KartothekDataset():
    """
    USAGE:
        d = KartothekDataset(path)
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

        print('-- read coadd table --')
        self.fetch_coadd_table()  # currently ignoring forced/unforced

        print('-- generate other metadata fields --')
        self.post_process_metadata()

        print('-- read visit data --')
        self.fetch_visits_by_metric()

        print('-- done with reads --')

    def get_coadd_ddf_by_filter_metric(self, filter_name, metrics,
                                       tracts, coadd_version='unforced'):

        for t in tracts:
            if t not in self.tracts:
                print('WARNING: Selected tract {} missing in data'.format(t))

        # filter out any tracts not in data
        valid_tracts = list(set(self.tracts).intersection(set(tracts)))

        if not valid_tracts:
            print('WARNING: No Valid tracts selected...using all tracts'.format(t))
            valid_tracts = self.tracts

        predicates = [[('tract', 'in', valid_tracts),
                       ('filter', '==', filter_name)]]

        dataset = "coadd_{}".format(coadd_version)

        columns = metrics + self.flags + ['ra', 'dec', 'filter',
                                          'psfMag', 'patch']

        store = partial(get_store_from_url, 'hfs://' + str(self.path))

        coadd_df = read_dataset_as_ddf(predicates=predicates,
                                       dataset_uuid=dataset,
                                       columns=columns,
                                       store=store,
                                       table='table').compute()

        # hack label in ...
        coadd_df['label'] = 'star'
        coadd_df.set_index('filter', inplace=True)

        return coadd_df

    def get_patch_count(self, filters, tracts, coadd_version='unforced'):

        return 1

        predicates = []

        if filters:
            predicates.append(('filter', 'in', filters))

        if tracts:
            predicates.append(('tract', 'in', tracts))

        dataset = "coadd_{}".format(coadd_version)

        columns = ['patch']

        store = partial(get_store_from_url, 'hfs://' + str(self.path))

        if predicates:

            coadd_df = read_dataset_as_ddf(predicates=[predicates],
                                           dataset_uuid=dataset,
                                           columns=columns,
                                           store=store,
                                           table='table')
        else:
            coadd_df = read_dataset_as_ddf(dataset_uuid=dataset,
                                           columns=columns,
                                           store=store,
                                           table='table')


        return coadd_df.drop_duplicates().count().compute()['patch']

    def parse_metadata_from_file(self):
        if self.path.joinpath(METADATA_FILENAME).exists():
            self.metadata_path = self.path.joinpath(METADATA_FILENAME)
        else:
            self.metadata_path = Path(os.environ.get('LSST_META', os.curdir)).joinpath(self.path.name, METADATA_FILENAME)

        with self.metadata_path.open('r') as f:
            self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
            self.failures = self.metadata.get('failures', {})
            if self.tracts is None:
                self.tracts = list(set(x for v in self.metadata['visits'].values() for x in v.keys()))

    def fetch_coadd_table(self, coadd_version='unforced'):
        table = 'qaDashboardCoaddTable'
        store = partial(get_store_from_url, 'hfs://' + str(self.path))
        print(str(self.path))
        predicates = [[('tract', 'in', self.tracts)]]
        dataset = "coadd_{}".format(coadd_version)

        coadd_df = read_dataset_as_ddf(predicates=predicates,
                                       dataset_uuid=dataset,
                                       store=store,
                                       table='table')

        # hack label in ...
        coadd_df['label'] = 'star'

        self.coadd[table] = coadd_df

    def post_process_metadata(self):
        df = self.coadd['qaDashboardCoaddTable']
        self.flags = df.columns[df.dtypes == bool].to_list()
        self.filters = list(self.metadata['visits'].keys())
        self.metrics = (set(df.columns.to_list()) -
                        set(self.flags) -
                        set(['patch', 'dec', 'label',
                             'psfMag', 'ra', 'filter',
                             'dataset', 'dir0', 'tract']))

    def fetch_visits(self):
        store = partial(get_store_from_url, 'hfs://' + str(self.path))
        predicates = [[('tract', 'in', self.tracts)]]
        self.visits = read_dataset_as_ddf(predicates=predicates,
                                          dataset_uuid='visits',
                                          store=store,
                                          table='table')

    def get_visits_by_metric_filter(self, filt, metric):

        store = partial(get_store_from_url, 'hfs://' + str(self.path))

        columns = ['filter', 'tract', 'visit', 'calib_psf_used',
                   'calib_psf_candidate', 'calib_photometry_reserved',
                   'qaBad_flag', 'ra', 'dec', 'psfMag'] + [metric]

        visits_ddf = read_dataset_as_ddf(dataset_uuid="visits",
                                         predicates=[[('filter', '==', filt)]],
                                         store=store,
                                         columns=columns,
                                         table='table')
        # hack label in ...
        visits_ddf['label'] = 'star'

        return visits_ddf[visits_ddf[metric].notnull()]

    def fetch_visits_by_metric(self):
        for filt in self.filters:
            self.visits_by_metric[filt] = {}
            for metric in self.metrics:
                try:
                    ddf = self.get_visits_by_metric_filter(filt, metric)
                except:
                    print('WARNING: problem loading visits for {} metric and {} filter'.format(metric, filt))
                    ddf = None

                self.visits_by_metric[filt][metric] = ddf
