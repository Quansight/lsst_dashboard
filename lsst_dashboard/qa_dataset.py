import logging
import re
from functools import partial

import numpy as np
import pandas as pd
import holoviews as hv

from lsst.qa.explorer.match import match_lists
from lsst.qa.explorer.plots import filter_dset, FilterStream


class QADataset(object):
    """Convenience wrapper of holoviews Dataset for catalog-type data

    The main purpose of this object is to easily package a `DataFrame`
    containing some catalog-style data (e.g., the result of a postprocessing
    analysis) as a Holoviews `Dataset` object.  This involves the following:

    * defining which columns are "key dimensions" ("kdims") or "value dimensions"
    ("vdims").  The `_idNames` and `_kdims` attributes define what columns
    get automatically categorized as "kdims" (such as RA, Dec, label, etc.); also, any
    flags (identified as boolean columns by inspection) are also kdims.  By default,
    everything else is a "vdim" (meaning, an quantity whose value is interesting).
    Otherwise, specific 'vdims' may be noted, in which case some columns will be
    unused by the `hv.Dataset`.

    * Defining the `hv.Dataset` object, which beyond defining the dimensions
    also necessitates getting rid of infs/nans in the data.

    Some subclasses of `QADataset` may also choose to defer creation of the `.df`
    attribute, which in this main object is passed as a parameter at initialization.
    Such a subclass will then need to define a `_makeDataFrame` method
    (See, for example, `MatchedQADataset`.)

    Parameters
    ----------
    df : `pandas.DataFrame`
        Dataframe to wrap; e.g., the output of a `PostprocessAnalysis`.

    vdims : `str` or `list`, optional
        Column names to count as "value dimensions"; that is,
        quantities that you might want to explore.  If 'all' (default),
        then the vdims will be everything except `._kdims`, `._idNames`
        and flags (which are defined as all boolean columns).

    """

    _idNames = ('patchId', 'tractId')
    _kdims = ('ra', 'dec', 'psfMag', 'label')

    def __init__(self, df, vdims='all'):

        self._df = df
        self._vdims = vdims

        self._ds = None
        self._flags = None

    @property
    def df(self):
        """Dataframe containing all columns to be explored

        This is either passed directly in the constructor or
        created by the `._makeDataFrame()` method, which
        can be implemented differently in subclasses.
        """
        if self._df is None:
            self._makeDataFrame()
        return self._df

    def _makeDataFrame(self):
        raise NotImplementedError('Must implement _makeDataFrame if df is not initialized.')

    @property
    def idNames(self):
        return [n for n in self._idNames if n in self.df.columns]

    @property
    def flags(self):
        """All boolean columns of dataframe
        """
        if self._flags is None:
            self._flags = [c for c in self.df.columns
                           if self.df[c].dtype == np.dtype('bool')]
        return self._flags

    def _getDims(self):
        """Construct kdims and vdims for hv.Dataset object
        """
        kdims = []
        vdims = []
        for c in self.df.columns:
            if (c in self._kdims or
                    c in self._idNames or
                    c in self.flags):
                kdims.append(c)
            else:
                if self._vdims == 'all':
                    vdims.append(c)
                elif c in self._vdims:
                    vdims.append(c)

        return kdims, vdims

    @property
    def vdims(self):
        _, vdims = self._getDims()
        return vdims

    @property
    def kdims(self):
        kdims, _ = self._getDims()
        return kdims

    @property
    def ds(self):
        """Holoviews Dataset object
        """
        if self._ds is None:
            self._makeDataset()
        return self._ds

    def _makeDataset(self):
        kdims, vdims = self._getDims()

        df = self.df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
        ds = hv.Dataset(df, kdims=kdims, vdims=vdims)
        self._ds = ds

    def skyPoints(self, vdim, maxMag, label='star', magCol='psfMag',
                  filter_range=None, flags=None, bad_flags=None):
        """Points object with ra, dec as key dimensions and requested value dimension

        This is used by `skyDmap` to make an interactive colormapped plot of
        individual points in the dataset (as opposed to a datashaded view.)

        Parameters
        ----------
        vdim : str
            Name of requested value dimension (one of the columns of `.df`);
            this will be the color_index.

        maxMag : float
            Faint limit of objects to plot.  Points are selected to have
            `magCol < maxMag`.

        label : str
            Label to select points (e.g., `'star'`, `'galaxy'` or any other
            label that is in the 'label' column of `.df`)

        magCol : str
            Dimension that gets filtered on by `maxMag`.

        filter_range, flags, bad_flags : `dict`, `list`, `list`
            Arguments passed by `FilterStream` object (from `skyDmap`).

        """
        selectDict = {magCol: (0, maxMag),
                      'label': label}
        ds = self.ds.select(**selectDict)
        ds = filter_dset(ds, filter_range=filter_range, flags=flags, bad_flags=bad_flags)

        pts = hv.Points(ds, kdims=['ra', 'dec'], vdims=self.vdims + [magCol] + self.idNames)
        return pts.options(color_index=vdim)

    def skyDmap(self, vdim, magRange=(np.arange(16, 24.1, 0.2)), magCol='psfMag',
                filter_stream=None,
                range_override=None):
            """Dynamic map of values of a particular dimension

            Parameters
            ----------
            vdim : str
                Name of dimension to explore.

            magRange : array
                Values of faint magnitude limit.  Only points up to this limit will be plotted.
                Beware of scrolling to too faint a limit; it might give you too many points!

            filter_stream : `qa.explorer.plots.FilterStream`, optional
                Stream of constraints that controls what data to display.  Useful to link
                multiple plots together

            range_override : (min, max), optional
                By default the colormap will be scaled between the 0.005 to 0.995 quantiles
                of the entire data (not just that displayed).  Sometimes this is not a useful range to view,
                so this parameter allows a custom colormap range to be set.
            """
            if filter_stream is not None:
                streams = [filter_stream]
            else:
                streams = [FilterStream()]
            fn = partial(QADataset.skyPoints, self=self, vdim=vdim, magCol=magCol)
            dmap = hv.DynamicMap(fn, kdims=['maxMag', 'label'],
                                 streams=streams)

            y_min = self.df[vdim].quantile(0.005)
            y_max = self.df[vdim].quantile(0.995)

            ra_min, ra_max = self.df.ra.quantile([0, 1])
            dec_min, dec_max = self.df.dec.quantile([0, 1])

            ranges = {vdim: (y_min, y_max),
                      'ra': (ra_min, ra_max),
                      'dec': (dec_min, dec_max)}
            if range_override is not None:
                ranges.update(range_override)

            dmap = dmap.redim.values(label=['galaxy', 'star'],
                                     maxMag=magRange).redim.range(**ranges)
            return dmap


class MatchedQADataset(QADataset):
    """A QADataset constructed from positional matching of two others

    The purpose of this is to, e.g., compare results of some postprocessing
    analysis of one rerun to that of another.  This does so by closest
    RA/Dec spatial matching.  For this object, the `.df` attribute
    is computed as follows:  all 'kdims' are taken from the `data1`
    `QADataset`, and the value of the 'vdims' is computed as the
    difference of the values between the datasets (`data2 - data1`).

    Matching is done using `lsst.qa.explorer.match.match_lists`, which
    uses a KDTree.

    TODO: Results should be cached in `match_registry`, if provided.

    Parameters
    ----------
    data1, data2 : `lsst.qa.explorer.dataset.QADataset`
        Two datasets to match.

    match_radius : `float`
        Max match distance in arcsec.  Default is 0.5.

    match_registry : `str` (optional)
        Path to an .h5 file containing cached match results.
        (Not implemented yet; implementation should parallel
        that of `lsst.qa.explorer.catalog.MatchedCatalog`).

    """

    def __init__(self, data1, data2,
                 match_radius=0.5, match_registry=None,
                 **kwargs):
        self.data1 = data1
        self.data2 = data2
        self.match_radius = match_radius
        self.match_registry = match_registry

        self._matched = False
        self._match_inds1 = None
        self._match_inds2 = None
        self._match_distance = None

        self._df = None
        self._ds = None

    def _match(self):
        if not all([c in self.data1.df.columns for c in ['ra', 'dec',
                                                         'detect_isPrimary']]):
            raise ValueError('Dataframes must have `detect_isPrimary` flag, ' +
                             'as well as ra/dec.')
        isPrimary1 = self.data1.df['detect_isPrimary']
        isPrimary2 = self.data2.df['detect_isPrimary']

        ra1, dec1 = self.data1.df.ra[isPrimary1], self.data1.df.dec[isPrimary1]
        ra2, dec2 = self.data2.df.ra[isPrimary2], self.data2.df.dec[isPrimary2]
        id1 = ra1.index
        id2 = ra2.index

        dist, inds = match_lists(ra1, dec1, ra2, dec2, self.match_radius/3600)

        good = np.isfinite(dist)

        fmtArgs = good.sum(), self.match_radius, (~good).sum()
        logging.info('{0} matched within {1} arcsec, {2} did not.'.format(*fmtArgs))

        # Save indices as labels, not positions, as required by dask
        i1 = id1[good]
        i2 = id2[inds[good]]
        d = pd.Series(dist[good] * 3600, index=id1[good], name='match_distance')

        self._match_inds1 = i1
        self._match_inds2 = i2
        self._match_distance = d

        self._matched = True

    @property
    def match_distance(self):
        """Distance between objects identified as matches
        """
        if self._match_distance is None:
            self._match()
        return self._match_distance

    @property
    def match_inds1(self):
        if self._match_inds1 is None:
            self._match()
        return self._match_inds1

    @property
    def match_inds2(self):
        if self._match_inds2 is None:
            self._match()
        return self._match_inds2

    @property
    def match_inds(self):
        return self.match_inds1, self.match_inds2

    def _combine_operation(self, v1, v2):
        return v2 - v1

    def _makeDataFrame(self):
        df1 = self.data1.df.copy()
        df2 = self.data2.df.copy()

        # For any *_magDiff columns, add back the psfMag (x) for a more useful difference
        for df in (df1, df2):
            for c in df.columns:
                m = re.search('(.+_mag)Diff$', c)
                if m:
                    newCol = m.group(1)
                    df[newCol] = df[c] + df['psfMag']

        id1, id2 = self.match_inds

        vdims = self.vdims
        v1 = df1.loc[id1, vdims]
        v2 = df2.loc[id2, vdims]
        v2.index = v1.index

        df = df1.copy()
        df[vdims] = self._combine_operation(v1, v2)
        df['match_distance'] = self.match_distance

        self._df = df

    @property
    def flags(self):
        return self.data1.flags

    def _getDims(self):
        kdims, vdims = self.data1._getDims()

        # Replace the *magDiff vdims with *mag
        magDiffDims = [dim for dim in vdims if re.search('(.+_mag)Diff$', dim)]
        magDims = [dim[:-4] for dim in magDiffDims]

        for d1, d2 in zip(magDiffDims, magDims):
            vdims.remove(d1)
            vdims.append(d2)

        vdims.append('match_distance')
        return kdims, vdims
