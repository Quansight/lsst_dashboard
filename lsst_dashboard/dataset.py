import yaml
from pathlib import Path
import dask.dataframe as dd
from dask import delayed
import pandas as pd
import os
import itertools

try:
    from lsst.daf.persistence import Butler
    from lsst.qa.explorer.functors import StarGalaxyLabeller, Magnitude, RAColumn, DecColumn, CompositeFunctor
    transforms = CompositeFunctor({'label': StarGalaxyLabeller(),
                          'psfMag': Magnitude('base_PsfFlux_instFlux'),
                          'ra': RAColumn(),
                          'dec': DecColumn()})
except ImportError:
    Butler = None
    StarGalaxyLabeller, Magnitude, RAColumn, DecColumn, CompositeFunctor = [None, None, None, None, None]


METADATA_FILENAME = 'metadata.yaml'

class Dataset():
    """
    USAGE:
        d = Dataset(path)
        d.connect()
        d.init_data()
    """
    def __init__(self, path):
        self.conn = None
        self.path = Path(path)
        self.tables = {}
        self.visits = {}
        self.tables_df = {}
        self.visits_df = {}
        self.metadata = {}
        self.filters = []
        self.metrics = []
        self.failures = {}
        self.flags = []
        self.tracts = []

    def connect(self):
        # search for metadata.yaml file
        # 1. Look in path directory i.e. '/project/tmorton/tickets/DM-20015/RC2_w18/metadata.yaml'
        # 2. Look for datafolder in current directory i.e. './RC2_w18/metadata.yaml'
        # 3. Look for datafolder in dir specified in LSST_META env variable i.e. /user/name/lsst_meta/RC2_w18/metadata.yaml'
        #    when LSST_META='/user/name/lsst_meta'
        if self.path.joinpath(METADATA_FILENAME).exists():
            self.metadata_path = self.path.joinpath(METADATA_FILENAME)
        else:
            self.metadata_path = Path(os.environ.get('LSST_META', os.curdir)).joinpath(self.path.name, METADATA_FILENAME)

        with self.metadata_path.open('r') as f:
            self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
            self.filters = self.metadata.get('filters')
            self.metrics = self.metadata.get('metrics')
            self.failures = self.metadata.get('failures')
            self.flags = self.metadata.get('flags')
            self.tracts = self.metadata.get('tracts')

        # if Butler is available use it to connect. If not available we are reading from disk
        if Butler: #
            self.conn = Butler(str(self.path))

    def init_data(self):
        if self.conn is None:
            print('Butler not found, loading data from parquet')
            self.read_parquet()
            return

        print('loading coadd_forced table')
        self.fetch_coadd_table('forced')
        print('loading coadd_unforced table')
        self.fetch_coadd_table('unforced')
        for filt in self.filters:
            print(f'loading visit data for filter {filt}')
            self.fetch_visits(filt)

    def fetch_coadd_table(self, version='unforced'):
        table = 'analysisCoaddTable_' + version
        dfs = []
        for filt in self.filters:
            for tract in self.tracts:
                dfs.append(delayed(self._load_coadd_table)(table, filt, tract))

        self.tables_df[table] = dd.from_delayed(dfs)

    def fetch_visits(self, filt):
        visits = []
        for tract in self.tracts:
            df = self.conn.get('visitMatchTable', tract=int(tract), filter=filt).toDataFrame()
            visits.append([delayed(self._fetch_visit)(visit, tract, filt) for visit in df['matchId'].columns])

        self.visits_df[filt] = dd.from_delayed(list(itertools.chain(*visits)))

    def _load_coadd_table(self, table, filt, tract):
        df = self.conn.get(table, tract=int(tract), filter=filt)
        new_cols = transforms(df)
        cols = self.metrics + self.flags + ['patchId', 'id']
        df = pd.concat([df.toDataFrame(columns=cols), new_cols], axis=1)
        df['filter'] = filt
        df['tract'] = tract
        return df

    def _fetch_visit(self, visit, tract, filt):
        df = self.conn.get('analysisVisitTable', visit=int(visit), tract=int(tract), filter=filt)
        df = df.toDataFrame(self.metrics + self.flags)
        df['visit'] = visit
        df['tract'] = tract
        df['filt'] = filt
        return df

    def read_parquet(self):
        p = self.path
        for table in ['analysisCoaddTable_forced', 'analysisCoaddTable_unforced']:
            self.tables_df[table] = dd.read_parquet(p.joinpath(table), engine='pyarrow')

        for filt in self.filters:
            self.visits_df[filt] = dd.read_parquet(p.joinpath(f'{filt}_visits'), engine='pyarrow')

    def to_parquet(self, path):
        p = Path(path)
        for table in ['analysisCoaddTable_forced', 'analysisCoaddTable_unforced']:
            self.tables[table].to_parquet(p.joinpath(table), engine='pyarrow', compression='snappy')

        for filt in self.filters:
            self.visits[filt].to_parquet(p.joinpath(f'{filt}_visits'), engine='pyarrow', compression='snappy')

    def load_from_hdf(self):
        tables = {}
        visits = {}
        for f in self.path.glob('*.h5'):
            filt = f.name.split('.')[0]
            tables[filt] = {}
            visits[filt] = {}
            with pd.HDFStore(f) as hdf:
                for k in hdf.keys():
                    if k.endswith('meta'):
                        continue
                    table, tract = k.split('/')[-1].rsplit('_', 1)
                    if table=='visits':
                        visits[filt][tract] = hdf.select(k)
                    if table not in tables[filt]:
                        tables[filt][table] = {}
                        tables[filt][table][tract] = hdf.select(k)

        self.tables = tables
        self.visits = visits
