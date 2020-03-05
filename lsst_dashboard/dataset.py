import yaml
from pathlib import Path
import dask.dataframe as dd
from kartothek.io.dask.dataframe import update_dataset_from_ddf, read_dataset_as_ddf
from storefact import get_store_from_url
from functools import partial
import os

try:
    from lsst.daf.persistence import Butler
except:
    Butler = None

METADATA_FILENAME = 'dashboard_metadata.yaml'

class Dataset():
    """
    USAGE:
        d = Dataset(path)
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
        self.store = None
        self.connect()

    def connect(self):
        # search for metadata.yaml file
        # 1. Look in path directory i.e. '/project/tmorton/tickets/DM-20015/RC2_w18/metadata.yaml'
        # 2. Look for datafolder in current directory i.e. './RC2_w18/metadata.yaml'
        # 3. Look for datafolder in dir specified in LSST_META env variable i.e. /user/name/lsst_meta/RC2_w18/metadata.yaml'
        #    when LSST_META='/user/name/lsst_meta'
      
        print('-- read metadata file --')

        # if Butler is available use it to connect. If not available we are reading from disk
        if Butler:
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
                print(f'{self.path} is not available in Butler attempting to read parquet files instead')
        
        if self.conn is None:
            if self.path.joinpath(METADATA_FILENAME).exists():
                self.metadata_path = self.path.joinpath(METADATA_FILENAME)
            else:
                self.metadata_path = Path(os.environ.get('LSST_META', os.curdir)).joinpath(self.path.name, METADATA_FILENAME)

            with self.metadata_path.open('r') as f:
                self.metadata = yaml.load(f, Loader=yaml.SafeLoader)
                self.failures = self.metadata.get('failures', {})
                if self.tracts is None:
                    self.tracts = list(set(x for v in self.metadata['visits'].values() for x in v.keys())) 

        # read kartothek partitioned data store
        self.store = partial(get_store_from_url, 'hfs://' + self.metadata['data_path'])
        self.fetch_coadds_view = partial(
            read_dataset_as_ddf,
            dataset_uuid="coadds",
            store=self.store,
            table='table'
        )
        self.fetch_visits_view = partial(
            read_dataset_as_ddf,
            dataset_uuid="visits",
            store=self.store,
            table='table'
        )


def repartition_dataset(coadd_df, visits_df, output_path):
    """Repartition dataset using kartothek

    currently tested with the following dataframes

    coadd_df = dd.read_parquet('/project/dharhas/DM-21335-December/*CoaddTable*')
    visits_df = dd.read_parquet('/project/dharhas/DM-21335-December/*VisitTable*')

    """    
    store_factory = partial(get_store_from_url, 'hfs://' + output_path)

    # remove problematic 'label' column
    if 'label' in coadd_df.columns:
        del coadd_df.drop('label', axis=1)['label']

    if 'label' in visits_df.columns:
        del visits_df['label']

    # convert to categories to save space and make queries more efficient
    coadd_df = coadd_df.categorize(columns=['filter', 'tractId', 'dataset'])
    visits_df = visits_df.categorize(columns=['filter', 'tractId'])

    #rename columns to what the dashboard expects
    coadd_df = coadd_df.rename({'patchId': 'patch', 'tractId': 'tract'})
    visits_df = visits_df.rename({'tractId': 'tract', 'visitId': 'visit'})

    graph = update_dataset_from_ddf(
        coadd_df,
        dataset_uuid="coadds",
        store=store_factory,
        table='table',
        shuffle=True,
        partition_on=['dataset', 'filter', 'tractId'],
    )
    graph.compute()

    graph = update_dataset_from_ddf(
        visits_df,
        dataset_uuid="visits",
        store=store_factory,
        table='table',
        shuffle=True,
        partition_on=['filter', 'tractId'],
    )
    graph.compute()
    
    return store_factory