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
        self.path = Path(path)
        self.tables = {}
        self.visits = {}
    
    def connect(self):
        # search for metadata file
        if self.path.joinpath(METADATA_FILENAME).exists():
            self.metadata_path = self.path.joinpath(METADATA_FILENAME)
        else: 
            self.metadata_path = Path(os.environ.get('LSST_META', os.curdir)).joinpath(self.path.name + f'_{METADATA_FILENAME}')

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

    def get_table(self, table, tract, filt):
        if self.conn:
            df = self.conn.get(table, tract=int(tract), filter=filt)
        else: 
            raise NotImplementedError

        return df

    def init_data(self):
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
        
        self.tables[table] = dd.from_delayed(dfs)

    def fetch_visits(self, filt):
        visits = []
        for tract in self.tracts:
            df = self.get_table('visitMatchTable', tract, filt).toDataFrame()
            visits.append([delayed(self._fetch_visit)(visit, tract, filt) for visit in df['matchId'].columns])
                    
        self.visits[filt] = dd.from_delayed(list(itertools.chain(*visits)))

    def _load_coadd_table(self, table, filt, tract):
        df = self.get_table(table, tract, filt)
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

    def write_tables(self, path, filt, sample=None):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        h5_file = p.joinpath(f'{filt}.h5')

        for table, v1 in self.tables[filt].items():
            for tract in v1.keys():
                df = self.tables[filt][table][tract]
                if sample:
                    df = df.sample(sample)
                df.to_hdf(h5_file, key=f'{table}_{tract}', format='t')

        for tract in self.visits[filt].keys():
            self.visits[filt][tract].to_hdf(h5_file, f'visits_{tract}')

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

        