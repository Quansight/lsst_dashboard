import yaml
from pathlib import Path
import dask.dataframe as dd
import os

try:
    from lsst.daf.persistence import Butler
except:
    Butler = None

METADATA_FILENAME = 'metadata.yaml'

class Dataset():
    """
    USAGE:
        d = Dataset(path)
        d.connect()
        d.init_data()
    """
    def __init__(self, path, tracts=None):
        self.conn = None
        self.path = Path(path)
        self.coadd = {}
        self.visits = None
        self.visits_by_metric = {}
        self.metadata = {}
        self.filters = []
        self.metrics = []
        self.failures = {}
        self.flags = []
        self.tracts = tracts

    def connect(self):
        # search for metadata.yaml file
        # 1. Look in path directory i.e. '/project/tmorton/tickets/DM-20015/RC2_w18/metadata.yaml'
        # 2. Look for datafolder in current directory i.e. './RC2_w18/metadata.yaml'
        # 3. Look for datafolder in dir specified in LSST_META env variable i.e. /user/name/lsst_meta/RC2_w18/metadata.yaml'
        #    when LSST_META='/user/name/lsst_meta'
      
        print('-- read metadata file --')
        if self.path.joinpath(METADATA_FILENAME).exists():
            self.metadata_path = self.path.joinpath(METADATA_FILENAME)
        else:
            self.metadata_path = Path(os.environ.get('LSST_META', os.curdir)).joinpath(self.path.name, METADATA_FILENAME)

        with self.metadata_path.open('r') as f:
            self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
            self.failures = self.metadata.get('failures')
            if self.tracts is None:
                self.tracts = self.metadata.get('tracts')

        # if Butler is available use it to connect. If not available we are reading from disk
        if Butler:
            try:
                print('-- connect to butler --')
                self.conn = Butler(str(self.path))
            except:
                print(f'{self.path} is not available in Butler attempting to read parquet files instead')

        print('-- read coadd table --')
        self.fetch_coadd_table()  # currently ignoring forced/unforced
        # update metadata based on coadd table fields
        print('-- generate other metadata fields --')
        df = self.coadd['qaDashboardCoaddTable']
        self.flags = df.columns[df.dtypes == bool].to_list()
        self.filters = df['filter'].unique().compute().to_list() # this takes some time, mightbe better to read from metadata file
        self.metrics = set(df.columns.to_list()) - set(self.flags) - set(['patch', 'dec', 'label', 'psfMag', 'ra', 'filter', 'dataset', 'tract'])
        print('-- read visit data --')
        self.fetch_visits()
        self.fetch_visits_by_metric()
        print('-- done with reads --')

    def fetch_coadd_table(self, coadd_version='unforced'):
        table = 'qaDashboardCoaddTable'  # + coadd_version
        if self.conn:
            filenames = [self.conn.get(table, tract=int(t)).filename for t in self.tracts]
        else:
            filenames = [str(self.path.join(f'{table}-{t}.parq')) for t in self.tracts]

        # workaround for tract not reliably being in file:
        dfs = []
        for tract, f in zip(self.tracts, filenames):
            df = dd.read_parquet(f, npartitions=4).rename(columns={'patchId': 'patch'})
            df['tract'] = tract
            dfs.append(df)
        self.coadd[table] = dd.concat(dfs)

    def fetch_visits(self):
        table = 'qaDashboardVisitTable'
        if self.conn:
            filenames = [self.conn.get(table, tract=int(t)).filename for t in self.tracts]
        else:
            filenames = [str(self.path.join(f'{table}-{t}.parq')) for t in self.tracts]
        
        self.visits = dd.read_parquet(filenames, npartitions=16).rename(columns={'tractId': 'tract', 'visitId': 'visit', 'patchId': 'patch'})

    def fetch_visits_by_metric(self):
        cols = self.visits.columns[self.visits.dtypes == bool].to_list() + ['dec', 'label', 'psfMag', 'ra', 'filter', 'tract', 'visit']
        for filt in self.filters:
            self.visits_by_metric[filt] = {}
            for metric in self.metrics:
                visit_data = None
                if metric in self.visits.columns:
                    visit_data = self.visits[(self.visits.filter==filt)][[metric] + cols] 
                self.visits_by_metric[filt][metric] = visit_data