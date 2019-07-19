"collection of utilities for reading data from lsst butler and restructuring for plotting"
from pathlib import Path
from lsst.daf.persistence import butler

butler = None

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


def connect(path=None):
    """Connect to butler and store connection as global
    """
    global butler

    if path:
        butler = Butler(path)

    if butler is None:
        raise(ValueError('No existing butler connection found, please specify data path in get_connection(path)'))

    return butler


def scan_folder(path):
    """Given a folder return available tracts and filters
    """
    folder = Path(path)
    tracts = list(set([t.name.split('-')[-1] for t in folder.rglob('*tract*')]))
    filters = [f.name for f in folder.joinpath('plots').iterdir() if f.is_dir()]

    return tracts, filters


def get_metric(table, tract, filter, *, parquet=None):
    return pandas.concat([
        parquet.toDataFrame(columns=__annotations__['metric']+__annotations__['flag']), lsst.qa.explorer.functors.CompositeFunctor(composite)(parquet)
    ], axis=1).set_index(list(composite.keys())+__annotations__['flag'], append=True)

def get_coadd(tract, filter, *, metrics=None, flags=None, version='forced', parquet=None): 
    if metrics is None:
        metrics = _default_metrics()

    if flags is None:
        flags = _default_flags()

    if parquet is None: 
        parquet = connect().get(f'analysisCoaddTable_{version}', tract=tract, filter=filter)
        
    return parquet.toDataFrame()*metrics)

def get_match(tract, filter, *, parquet=conn.get('visitMatchTable', tract=tract, filter=filter)): return parquet.toDataFrame()['matchId']
    
def get_tables(table, tract, filter): return {'df': await get_metric(table, tract, filter), 'match': await get_match(tract, filter)}

def get_visit(visit, tract, filter, metric): return conn.get('analysisVisitTable', visit=visit, tract=tract, filter=filter).toDataFrame(metric)
    
def get_visits(visits, tract, filter, metric):
    with pandas.option_context('mode.use_inf_as_na', True): return {
        visit: await(get_visit(visit, tract, filter, metric)) for visit in visits}

def select(df, metric):
    self = pandas.DataFrame(df[metric].values, index=pandas.MultiIndex.from_arrays([df.index.get_level_values(object) for object in "psf ra dec".split()]), columns=metric)
        with pandas.option_context('mode.use_inf_as_na', True):  return self[metric]