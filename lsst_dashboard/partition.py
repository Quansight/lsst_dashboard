# repartition data using kartothe
from pathlib import Path
from dask import delayed
from kartothek.io.dask.dataframe import update_dataset_from_ddf, read_dataset_as_ddf
from storefact import get_store_from_url
from functools import partial
import pandas as pd
import dask.dataframe as dd
import dask.array as da


def getMetrics():
    return ['base_Footprint_nPix', 
            'Gaussian-PSF_magDiff_mmag',
            'CircAper12pix-PSF_magDiff_mmag',
            'Kron-PSF_magDiff_mmag',
            'CModel-PSF_magDiff_mmag',
            'traceSdss_pixel',
            'traceSdss_fwhm_pixel',
            'psfTraceSdssDiff_percent',
            'e1ResidsSdss_milli',
            'e2ResidsSdss_milli',
            'deconvMoments',
            'compareUnforced_Gaussian_magDiff_mmag',
            'compareUnforced_CircAper12pix_magDiff_mmag',
            'compareUnforced_Kron_magDiff_mmag',
            'compareUnforced_CModel_magDiff_mmag',
            'traceSdss_pixel',
            'traceSdss_fwhm_pixel',
            'psfTraceSdssDiff_percent',
            'e1ResidsSdss_milli',
            'e2ResidsSdss_milli',
            'base_PsfFlux_instFlux',
            'base_PsfFlux_instFluxErr'
    ]


def getFlags():
    return ['calib_psf_used',
            'calib_psf_candidate',
            'calib_photometry_reserved',
            'merge_measurement_i2',
            'merge_measurement_i',
            'merge_measurement_r2',
            'merge_measurement_r',
            'merge_measurement_z',
            'merge_measurement_y',
            'merge_measurement_g',
            'merge_measurement_N921',
            'merge_measurement_N816',
            'merge_measurement_N1010',
            'merge_measurement_N387',
            'merge_measurement_N515',
            'qaBad_flag'
    ]


def normalize(*args):
    """
    extract tract, visit ints from strings
    """
    return [int(x.split('-')[-1]) for x in args]


def normalize_df(df, sample_frac=None):
    """
    append ra, dec, psfMag to dataframe and cleanup
    """
    if sample_frac:
        df = df.sample(frac=sample_frac))
        
    df = df.assign(
        ra = da.rad2deg(df.coord_ra),
        dec = da.rad2deg(df.coord_dec),
        psfMag = -2.5 * da.log10(df.base_PsfFlux_instFlux)
    )

    del df['coord_ra']
    del df['coord_dec']
    df = df.rename(columns={'patchId': 'patch', 'ccdId':'ccd'})
    df = df.categorize(columns=['filter', 'tract'])

    return df
    

def ktk_repartition(src, dst, sample_frac=None):
    """
    Read data formatted in the Directory Structure below and repartition to ktk:

    'analysisVisitTable': 'plots/%(filter)s/tract-%(tract)d/visit-%(visit)d/%(tract)d_%(visit)d.parq',
    'analysisVisitTable_commonZp': 'plots/%(filter)s/tract-%(tract)d/visit-%(visit)d/%(tract)d_%(visit)d_commonZp.parq',
    'analysisCoaddTable_forced': 'plots/%(filter)s/tract-%(tract)d/%(tract)d_forced.parq',
    'analysisCoaddTable_unforced': 'plots/%(filter)s/tract-%(tract)d/%(tract)d_unforced.parq',
    'analysisColorTable': 'plots/color/tract-%(tract)d/%(tract)d_color.parq',
    'visitMatchTable': 'plots/%(filter)s/tract-%(tract)d/%(tract)d_visit_match.parq',

    """
    p = Path(src)
    if p.stem != 'plots':
        p = p.joinpath('plots')
    
    store_factory = partial(get_store_from_url, 'hfs://' + dst)

    coadd_cols = list(set(getMetrics() + getFlags() + ['coord_ra', 'coord_dec', 'patchId']))
    visit_cols = coadd_cols + ['ccdId', 'filter', 'tract', 'visit']
    
    print(f'...constructing coadd_forced dataframe')
    dfs = []
    for f in p.glob('**/*_forced.parq'):
        *_, filt, tract, _ = f.parts
        tract, = normalize(tract)
        df = delayed(pd.read_parquet)(f, columns=coadd_cols)
        df = delayed(pd.DataFrame.assign)(df, filter=filt, tract=tract)
        dfs.append(df)
    dfs = normalize_df(dd.from_delayed(dfs), sample_frac)
    repartition_dataset(store_factory, dfs, 'coadd_forced', 'patch')
    del dfs

    print(f'...constructing coadd_unforced dataframe')
    dfs = []
    for f in p.glob('**/*_unforced.parq'):
        *_, filt, tract, _ = f.parts
        tract, = normalize(tract)
        df = delayed(pd.read_parquet)(f, columns=coadd_cols)
        df = delayed(pd.DataFrame.assign)(df, filter=filt, tract=tract)
        dfs.append(df)
    dfs = normalize_df(dd.from_delayed(dfs), sample_frac))
    repartition_dataset(store_factory, dfs, 'coadd_unforced', 'patch')
    del dfs

    print(f'...constructing visits dataframe')
    dfs = []
    for f in p.glob('**/visit-*/*[0-9].parq'):
        *_, filt, tract, visit, _ = f.parts
        tract, visit = normalize(tract, visit)
        df = delayed(pd.read_parquet)(f)
        df = delayed(pd.DataFrame.assign)(df, filter=filt, tract=tract, visit=visit)
        dfs.append(df)
    dfs = dd.from_delayed(dfs)
    dfs = dfs[list(set(visit_cols).intersection(dfs.columns))] # not all visits have all cols
    dfs = normalize_df(dfs, sample_frac))
    repartition_dataset(store_factory, dfs, 'visits', 'ccd')
    del dfs

    return store_factory


def repartition_dataset(store_factory, df, name, bucket_by):
    """Repartition dataset using kartothek
    """
    print(f'... ...ktk repartitioning {name}')
    graph = update_dataset_from_ddf(
        df,
        dataset_uuid=name,
        store=store_factory,
        table='table',
        shuffle=True,
        num_buckets=4,
        bucket_by=bucket_by,
        partition_on=['filter', 'tract'],
    )
    graph.compute()

