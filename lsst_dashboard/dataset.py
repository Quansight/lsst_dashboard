import yaml
from pathlib import Path
import pandas as pd

try:
    from lsst.daf.persistence import Butler
    from lsst.qa.explorer.functors import StarGalaxyLabeller, Magnitude, RAColumn, DecColumn, CompositeFunctor
except ImportError:
    Butler = None
    StarGalaxyLabeller, Magnitude, RAColumn, DecColumn, CompositeFunctor = [None, None, None, None, None]

METADATA_FILENAME = 'metadata.yaml'

funcs = CompositeFunctor({'label': StarGalaxyLabeller(), 
                          'psfMag': Magnitude('base_PsfFlux_instFlux'),
                          'ra': RAColumn(),
                          'dec': DecColumn()})


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

    def get_table(self, table, tract, filt, sample=None):
        if self.conn:
            df = self.conn.get(table, tract=int(tract), filter=filt)
        else: 
            raise NotImplementedError

        return df

    def fetch_tables(self, tables=None, tracts=None, filters=None, metrics=None, flags=None, sample=None):

        if tables is None:
            tables = self.metadata['tables']

        if tracts is None:
            tracts = self.metadata['tracts']

        if filters is None:
            filters = self.metadata['filters']

        if metrics is None:
            metrics = self.metadata['metrics']

        if flags is None:
            flags = self.metadata['flags']

        dataset = {}
        for filt in filters:
            dataset[filt] = {}
            for table in tables:
                dataset[filt][table] = {}
                for tract in tracts:
                    print(f'filt={filt}, table={table}, tract={tract}')
                    dataset[filt][table][tract] = self.get_table(table, tract, filt, sample)
                    if 'coadd' in table.lower():
                        df = dataset[filt][table][tract]
                        new_cols = funcs(df)
                        cols = metrics + flags + ['patchId', 'id']
                        df = pd.concat([df.toDataFrame(columns=cols), new_cols], axis=1)
                    else:
                        df = dataset[filt][table][tract].toDataFrame()

                    if sample:
                        df = df.sample(sample)
                    dataset[filt][table][tract] = df

        self.tables = dataset

    def fetch_visit(self, visit, tract, filt, sample=None):
        df = self.conn.get('analysisVisitTable', visit=int(visit), tract=int(tract), filter=filt).toDataFrame(self.metadata['metrics'])
        if sample:
            df = df.sample(sample)
        return df

    def fetch_visits(self, tracts, filters, sample=None):
        self.visits = {}
        for filt in filters:
            self.visits[filt] = {}
            for tract in tracts:
                print(f'filt={filt}, tract={tract}')
                visits = pd.concat({str(visit): self.fetch_visit(visit, tract, filt, sample) for visit in self.tables[filt]['visitMatchTable'][tract]['matchId'].columns.astype(str)})
                # leave this transform for later so we can save a simpler file
                # visits = visits.set_index(pd.MultiIndex.from_arrays([pd.CategoricalIndex(visits.index.get_level_values(0).astype(str), name='visit'), visits.index.get_level_values('id')]))
                self.visits[filt][tract] = visits

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



#def open_dataset(path):
#    
#    p = Path(path)
#    if p.exists():
#        dataset = Dataset(path)
#        dataset.connect()
        