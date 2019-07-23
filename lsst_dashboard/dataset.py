import yaml
from pathlib import Path
import pandas as pd

try:
    from lsst.daf.persistence import Butler
except ImportError:
    Butler = None

METADATA_FILENAME = 'metadata.yaml'

def _default_metrics():
    return [
        'base_Footprint_nPix', 'Gaussian-PSF_magDiff_mmag', 'CircAper12pix-PSF_magDiff_mmag', 
        'Kron-PSF_magDiff_mmag', 'CModel-PSF_magDiff_mmag', 'traceSdss_pixel', 'traceSdss_fwhm_pixel', 
        'psfTraceSdssDiff_percent', 'e1ResidsSdss_milli', 'e2ResidsSdss_milli', 'deconvMoments', 
        'compareUnforced_Gaussian_magDiff_mmag', 'compareUnforced_CircAper12pix_magDiff_mmag', 
        'compareUnforced_Kron_magDiff_mmag', 'compareUnforced_CModel_magDiff_mmag'
    ]

def _default_flags():
    return [
        'calib_psf_used', 'calib_psf_candidate', 'calib_photometry_reserved', 'merge_measurement_i2', 
        'merge_measurement_i', 'merge_measurement_r2', 'merge_measurement_r', 'merge_measurement_z', 
        'merge_measurement_y', 'merge_measurement_g', 'merge_measurement_N921', 'merge_measurement_N816', 
        'merge_measurement_N1010', 'merge_measurement_N387', 'merge_measurement_N515', 'qaBad_flag'
    ]

class Dataset():
    def __init__(self, path):
        self.path = Path(path)
        self.metadata_path = self.path.joinpath(METADATA_FILENAME)
        self.conn = None
        self.metadata = None
        self.tables = {}
    
    def connect(self):
        if not self.path.joinpath('metadata.yaml').exists(): # todo: remove this once we have metadata yaml files saved in the data folders
            self.metadata = {'metrics': _default_metrics(), 'flags': _default_flags()}
        else:
            with self.metadata.open('r') as f:
                self.metadata = json.load(f)

        # if Butler is available use it to connect. If not available we are reading from disk
        if Butler: # 
            self.conn = Butler(str(self.path))

    def get_table(self, table, tract, filt):
        if self.conn:
            return self.conn.get(table, tract=tract, filter=filt)
        else:
            return pd.read_parquet(self.path.joinpath(f'{table}_{tract}_{filt}.parq'))

    def fetch_tables(self, tables=None, tracts=None, filters=None, metrics=None):

        if tables is None:
            tables = self.metadata['tables']

        if tracts is None:
            tracts = self.metadata['tracts']

        if filters is None:
            filters = self.metadata['filters']

        if metrics is None:
            metrics = self.metadata['metrics']

        dataset = {}
        for filt in filters:
            dataset[filt] = {}
            for table in tables:
                dataset[filt][table] = {}
                for tract in tracts:
                    dataset[filt][table][tract] = self.get_table(table, tract, filt)
                    if 'coadd' in table.lower():
                        dataset[filt][table][tract] = dataset[filt][table][tract].toDataFrame(columns=metrics)
                    else:
                        dataset[filt][table][tract] = dataset[filt][table][tract].toDataFrame()

        self.tables = dataset

    def fetch_visit(self, visit, tract, filt):
        return self.conn.get('analysisVisitTable', visit=visit, tract=tract, filter=filt).toDataFrame(self.metadata['metrics'])

    def fetch_visits(self, tracts, filters):
        for tract in tracts:
            for filt in filters:
                print(f'tract={tract}, filt={filt}')
                visits = pd.concat({visit: self.fetch_visit(visit, tract, filt)for visit in self.tables[filt]['visitMatchTable'][tract]['matchId'].columns})
                visits = visits.set_index(pandas.MultiIndex.from_arrays([pandas.CategoricalIndex(visits.index.get_level_values(0).astype(str), name='visit'), visits.index.get_level_values('id')]))
                self.visits = visits


#def open_dataset(path):
#    
#    p = Path(path)
#    if p.exists():
#        dataset = Dataset(path)
#        dataset.connect()
        