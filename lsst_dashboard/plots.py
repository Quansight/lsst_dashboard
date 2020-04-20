# from profilehooks import profile
from functools import partial

import param
import panel as pn
import numpy as np
import pandas as pd
import holoviews as hv
import datashader as ds
import colorcet as cc
from sklearn.preprocessing import minmax_scale

from param import ParameterizedFunction
from param import ParamOverrides
from bokeh.palettes import Greys9
from holoviews import opts
from holoviews.core.operation import Operation
from holoviews.core.util import isfinite
from holoviews.operation.element import apply_when
from holoviews.streams import (
    BoundsXY, LinkedStream, PlotReset, PlotSize, RangeXY, Stream
)
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.plotting.util import process_cmap

from holoviews.operation.datashader import datashade
from holoviews.operation.datashader import shade
from holoviews.operation.datashader import dynspread
from holoviews.operation.datashader import rasterize
from holoviews.operation import decimate

from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

from datashader.colors import viridis

decimate.max_samples = 5000

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Stream class that stores filters for various Dimensions
class FilterStream(Stream):
    """
    Stream to apply arbitrary filtering on a Dataset.

    Many of the plotting functions accept a `FilterStream` object;
    the utility of this is that you can define a single `FilterStream`,
    and if you connect the same one to all your plots, then all of the
    selections/flag selections/etc. can be linked.

    See the demo notebooks for an example of usage.
    """

    filter_range = param.Dict(default={}, doc="""
        Ranges of parameters to select.""")
    flags = param.List(default=[], doc="""
        Flags to select.""")
    bad_flags = param.List(default=[], doc="""
        Flags to ignore""")


class FlagSetter(Stream):
    """Stream for setting flags

    Most useful in context of a parambokeh widget, e.g.:

        from explorer.plots import FlagSetter
        import parambokeh

        flag_setter = FlagSetter(filter_stream=filter_stream, flags=data.flags, bad_flags=data.flags)
        parambokeh.Widgets(flag_setter, callback=flag_setter.event, push=False, on_init=True)

    Where `filter_stream` has been previously defined and connected to other plots
    for which you want to see points with certain flags shown/hidden/etc.
    """
    flags = param.ListSelector(default=[], objects=[], doc="""
        Flags to select""")
    bad_flags = param.ListSelector(default=[], doc="""
        Flags to ignore""")

    def __init__(self, filter_stream, **kwargs):
        super(FlagSetter, self).__init__(**kwargs)
        self.filter_stream = filter_stream

    def event(self, **kwargs):
        self.filter_stream.event(**kwargs)

#######################################################################################


class filter_dset(Operation):
    """Process a dataset based on FilterStream state (filter_range, flags, bad_flags)

    This is used in many applications to define dynamically selected `holoviews.Dataset`
    objects.
    """
    filter_range = param.Dict(default={}, doc="""
        Dictionary of filter bounds.""")
    flags = param.List(default=[], doc="""
        Flags to select.""")
    bad_flags = param.List(default=[], doc="""
        Flags to ignore""")

    def _process(self, dset, key=None):
        filter_dict = {} if self.p.filter_range is None else self.p.filter_range.copy()
        filter_dict.update({f: True for f in self.p.flags})
        filter_dict.update({f: False for f in self.p.bad_flags})
        dset = dset.select(**filter_dict)
        return dset


# Define Operation that filters based on FilterStream state (which provides the filter_range)
class filterpoints(Operation):
    """Process a dataset based on FilterStream state (filter_range, flags, bad_flags)

    This is used in many applications to define dynamically selected `holoviews.Points`
    objects.
    """

    filter_range = param.Dict(default={}, doc="""
        Dictionary of filter bounds.""")
    flags = param.List(default=[], doc="""
        Flags to select.""")
    bad_flags = param.List(default=[], doc="""
        Flags to ignore""")
    xdim = param.String(default='x', doc="Name of x-dimension")
    ydim = param.String(default='y', doc="Name of y-dimension")
    set_title = param.Boolean(default=False)

    def _process(self, dset, key=None):
        dset = filter_dset(dset, flags=self.p.flags, bad_flags=self.p.bad_flags,
                           filter_range=self.p.filter_range)
        kdims = [dset.get_dimension(self.p.xdim), dset.get_dimension(self.p.ydim)]
        vdims = [dim for dim in dset.dimensions() if dim.name not in kdims]
        pts = hv.Points(dset, kdims=kdims, vdims=vdims)
        if self.p.set_title:
            ydata = dset.data[self.p.ydim]
            title = 'mean = {:.3f}, std = {:.3f} ({:.0f})'.format(ydata.mean(),
                                                                  ydata.std(),
                                                                  len(ydata))
            pts = pts.relabel(title)
        return pts


class summary_table(Operation):
    ydim = param.String(default=None)
    filter_range = param.Dict(default={}, doc="""
        Dictionary of filter bounds.""")
    flags = param.List(default=[], doc="""
        Flags to select.""")
    bad_flags = param.List(default=[], doc="""
        Flags to ignore""")

    def _process(self, dset, key=None):

        ds = filter_dset(dset, filter_range=self.p.filter_range,
                         flags=self.p.flags, bad_flags=self.p.bad_flags)
        if self.p.ydim is None:
            cols = [dim.name for dim in dset.vdims]
        else:
            cols = [self.p.ydim]
        df = ds.data[cols]
        return hv.Table(df.describe().loc[['count', 'mean', 'std']])


def notify_stream(bounds, filter_stream, xdim, ydim):
    """
    Function to attach to bounds stream as subscriber to notify FilterStream.
    """
    logger.info('.notify_stream')
    l, b, r, t = bounds
    filter_range = dict(filter_stream.filter_range)
    for dim, (low, high) in [(xdim, (l, r)), (ydim, (b, t))]:
        # If you want to take the intersection of x selections, e.g.
        # if dim in filter_range:
        #     old_low, old_high = filter_range[dim]
        #     filter_range[dim]= (max(old_low, low), min(old_high, high))
        # else:
        #     filter_range[dim] = (low, high)
        filter_range[dim] = (low, high)
    filter_stream.event(filter_range=filter_range)


def reset_hook(plot, element, x_range=None, y_range=None):
    if x_range:
        plot.handles['x_range'].reset_start = x_range[0]
        plot.handles['x_range'].reset_end = x_range[1]
    if y_range:
        plot.handles['y_range'].reset_start = y_range[0]
        plot.handles['y_range'].reset_end = y_range[1]


def reset_stream(filter_stream, range_streams, resetting=True):
    if filter_stream:
        filter_stream.event(filter_range={}, flags=[], bad_flags=[])
    for range_stream in range_streams:
        range_stream.event(x_range=None, y_range=None)


def _link(streams, **contents):
    for stream in streams:
        if contents != stream.contents:
            stream.event(**contents)


def link_streams(*streams):
    """
    Links multiple streams of the same type.
    """
    assert len(set(type(s) for s in streams)) == 1
    for stream in streams:
        stream.add_subscriber(partial(_link, streams))
    if streams:
        _link(streams, **streams[0].contents)


class scattersky(ParameterizedFunction):
    """
    Creates two datashaded views from a Dataset.

    First plot is an x-y scatter plot, with colormap according to density
    of points; second plot is a sky plot where the colormap corresponds
    to the average y values of the first plot in each datashaded pixel.
    """

    xdim = param.String(default='x', doc="""
        Dimension of the dataset to use as x-coordinate""")

    ydim = param.String(default='y0', doc="""
        Dimension of the dataset to use as y-coordinate""")

    ra_sampling = param.Integer(default=None, doc="""
        How densely to sample the rasterized plot along the x-axis.""")

    dec_sampling = param.Integer(default=None, doc="""
        How densely to sample the rasterized plot along the x-axis.""")

    x_sampling = param.Integer(default=5000, doc="""
        How densely to sample the rasterized plot along the x-axis.""")

    y_sampling = param.Integer(default=5000, doc="""
        How densely to sample the rasterized plot along the y-axis.""")

    max_points = param.Integer(default=10000, doc="""
        Maximum number of points to display before switching to rasterize.""")

    scatter_cmap = param.String(default='fire', doc="""
        Colormap to use for the scatter plot""")

    sky_cmap = param.String(default='coolwarm', doc="""
        Colormap to use for the sky plot""")

    filter_stream = param.ClassSelector(default=FilterStream(), class_=FilterStream, doc="""
        Stream to which selection ranges get added.""")

    show_rawsky = param.Boolean(default=False, doc="""
        Whether to show the "unselected" sky points in greyscale when
        there is a selection.""")

    show_table = param.Boolean(default=False, doc="""
        Whether to show the table next to the plots.""")

    sky_range_stream = param.ClassSelector(default=None, class_=RangeXY)

    scatter_range_stream = param.ClassSelector(default=None, class_=RangeXY)

    # @profile(immediate=True)
    def __call__(self, dset, **params):
        self.p = ParamOverrides(self, params)
        if self.p.xdim not in dset.dimensions():
            raise ValueError('{} not in Dataset.'.format(self.p.xdim))
        if self.p.ydim not in dset.dimensions():
            raise ValueError('{} not in Dataset.'.format(self.p.ydim))
        if ('ra' not in dset.dimensions()) or ('dec' not in dset.dimensions()):
            raise ValueError('ra and/or dec not in Dataset.')

        # Compute sampling
        ra_range = (ra0, ra1) = dset.range('ra')
        if self.p.ra_sampling:
            ra_sampling = (ra1-ra0)/self.p.xsampling
        else:
            ra_sampling = None

        dec_range = (dec0, dec1) = dset.range('dec')
        if self.p.dec_sampling:
            dec_sampling = (dec1-dec0)/self.p.ysampling
        else:
            dec_sampling = None

        x_range = (x0, x1) = dset.range(self.p.xdim)
        if self.p.x_sampling:
            x_sampling = (x1-x0)/self.p.x_sampling
        else:
            x_sampling = None

        y_range = (y0, y1) = dset.range(self.p.ydim)
        if self.p.y_sampling:
            y_sampling = (y1-y0)/self.p.y_sampling
        else:
            y_sampling = None

        # Set up scatter plot
        scatter_range = RangeXY()
        if self.p.scatter_range_stream:
            def redim_scatter(dset, x_range, y_range):
                ranges = {}
                if x_range and all(isfinite(v) for v in x_range):
                    ranges[self.p.xdim] = x_range
                if y_range and all(isfinite(v) for v in x_range):
                    ranges[self.p.ydim] = y_range
                return dset.redim.range(**ranges) if ranges else dset
            dset_scatter = dset.apply(redim_scatter, streams=[self.p.scatter_range_stream])
            link_streams(self.p.scatter_range_stream, scatter_range)
        else:
            dset_scatter = dset
        scatter_pts = dset_scatter.apply(
            filterpoints, streams=[self.p.filter_stream],
            xdim=self.p.xdim, ydim=self.p.ydim
        )
        scatter_streams = [scatter_range, PlotSize()]
        scatter_rasterize = rasterize.instance(
            streams=scatter_streams, x_sampling=x_sampling,
            y_sampling=y_sampling
        )
        cmap = process_cmap(self.p.scatter_cmap)[:250] if self.p.scatter_cmap == 'fire' else self.p.scatter_cmap
        scatter_rasterized = apply_when(
            scatter_pts, operation=scatter_rasterize,
            predicate=lambda pts: len(pts) > self.p.max_points
        ).opts(
            opts.Image(clim=(1, np.nan), clipping_colors={'min': 'transparent'},
                       cmap=cmap),
            opts.Points(clim=(1, np.nan), clipping_colors={'min': 'transparent'},
                        cmap=cmap),
            opts.Overlay(hooks=[partial(reset_hook, x_range=x_range, y_range=y_range)])
        )

        # Set up sky plot
        sky_range = RangeXY()
        if self.p.sky_range_stream:
            def redim_sky(dset, x_range, y_range):
                ranges = {}
                if x_range and all(isfinite(v) for v in x_range):
                    ranges['ra'] = x_range
                if y_range and all(isfinite(v) for v in x_range):
                    ranges['dec'] = y_range
                return dset.redim.range(**ranges) if ranges else dset
            dset_sky = dset.apply(redim_sky, streams=[self.p.sky_range_stream])
            link_streams(self.p.sky_range_stream, sky_range)
        else:
            dset_sky = dset
        sky_pts = dset_sky.apply(
            filterpoints, xdim='ra', ydim='dec', set_title=False,
            streams=[self.p.filter_stream]
        )
        skyplot_streams = [sky_range, PlotSize()]
        sky_rasterize = rasterize.instance(
            aggregator=ds.mean(self.p.ydim), streams=skyplot_streams,
            x_sampling=ra_sampling, y_sampling=dec_sampling
        )
        sky_rasterized = apply_when(
            sky_pts, operation=sky_rasterize,
            predicate=lambda pts: len(pts) > self.p.max_points
        ).opts(
            opts.Image(bgcolor="black", cmap=self.p.sky_cmap, symmetric=True),
            opts.Points(bgcolor="black", cmap=self.p.sky_cmap, symmetric=True),
            opts.Overlay(hooks=[partial(reset_hook, x_range=ra_range,
                                        y_range=dec_range)])
        )

        # Set up BoundsXY streams to listen to box_select events and notify FilterStream
        scatter_select = BoundsXY(source=scatter_pts)
        scatter_notifier = partial(notify_stream, filter_stream=self.p.filter_stream,
                                   xdim=self.p.xdim, ydim=self.p.ydim)
        scatter_select.add_subscriber(scatter_notifier)

        sky_select = BoundsXY(source=sky_pts)
        sky_notifier = partial(notify_stream, filter_stream=self.p.filter_stream,
                               xdim='ra', ydim='dec')
        sky_select.add_subscriber(sky_notifier)

        # Reset
        reset = PlotReset(source=sky_pts)
        reset.add_subscriber(partial(reset_stream, self.p.filter_stream,
                                     [self.p.sky_range_stream,
                                      self.p.scatter_range_stream]))

        raw_scatterpts = filterpoints(dset, xdim=self.p.xdim, ydim=self.p.ydim)
        raw_scatter = datashade(
            raw_scatterpts, cmap=list(Greys9[::-1][:5]), streams=scatter_streams,
            x_sampling=x_sampling, y_sampling=y_sampling
        )
        scatter_p = (raw_scatter*scatter_rasterized)

        if self.p.show_rawsky:
            raw_skypts = filterpoints(dset, xdim=self.p.xdim, ydim=self.p.ydim)
            raw_sky = datashade(
                rawskypts, cmap=list(Greys9[::-1][:5]), streams=skyplot_streams,
                x_sampling=ra_sampling, y_sampling=dec_sampling
            )
            sky_p = raw_sky*sky_rasterized
        else:
            sky_p = sky_rasterized

        if self.p.show_table:
            table = dset.apply(summary_table, ydim=self.p.ydim, streams=[self.p.filter_stream])
            table = table.opts()
            layout = (table + scatter_p + sky_p)
        else:
            layout = (scatter_p + sky_p).opts(sizing_mode='stretch_width')

        return layout.opts(
            opts.Image(colorbar=True, responsive=True,
                       tools=['box_select', 'hover']),
            opts.Layout(sizing_mode='stretch_width'),
            opts.Points(color=self.p.ydim, tools=['hover']),
            opts.RGB(alpha=0.5),
            opts.Table(width=200)
        )


class multi_scattersky(ParameterizedFunction):
    """Layout of multiple scattersky plots, one for each vdim in dset
    """

    filter_stream = param.ClassSelector(default=FilterStream(), class_=FilterStream)

    height = param.Number(default=300)
    width = param.Number(default=900)
    xdim = param.String(default='x', doc="""
        Dimension of the dataset to use as x-coordinate""")

    def _get_ydims(self, dset):
        # Get dimensions from first Dataset type found in input
        return [dim.name for dim in dset.traverse(lambda x: x, [hv.Dataset])[0].vdims]

    def __call__(self, dset, **params):
        self.p = param.ParamOverrides(self, params)
        return hv.Layout([scattersky(dset, filter_stream=self.p.filter_stream,
                                     xdim=self.p.xdim, ydim=ydim,
                                     height=self.p.height, width=self.p.width)
                          for ydim in self._get_ydims(dset)]).cols(3).opts(merge_tools=False)


class skypoints(Operation):
    """Creates Points with ra, dec as kdims, and interesting stuff as vdims
    """
    filter_range = param.Dict(default={}, doc="""
        Dictionary of filter bounds.""")
    flags = param.List(default=[], doc="""
        Flags to select.""")
    bad_flags = param.List(default=[], doc="""
        Flags to ignore""")

    def _process(self, dset, key=None):
        dset = filter_dset(dset, filter_range=self.p.filter_range,
                           flags=self.p.flags, bad_flags=self.p.bad_flags)
        return hv.Points(dset, [dset.get_dimension('ra'), dset.get_dimension('dec')],
                         dset.vdims+[dset.get_dimension('label')]).opts(responsive=True)


class skyplot(ParameterizedFunction):
    """Skyplot of RA/dec switching between rasterized and raw data view.

    Colormapped by a third dimension.
    """

    aggregator = param.ObjectSelector(default='mean', objects=['mean', 'std', 'count'], doc="""
        Aggregator for datashading.""")

    cmap = param.String(default='coolwarm', doc="""
        Colormap to use.""")

    decimate_size = param.Number(default=5, doc="""
        Size of (invisible) decimated points.""")

    max_points = param.Integer(default=10000, doc="""
        Maximum number of points to display before switching to rasterize.""")

    vdim = param.String(default=None, doc="""
        Dimension to use for colormap.""")

    ra_sampling = param.Integer(default=None, doc="""
        How densely to sample the rasterized plot along the x-axis.""")

    dec_sampling = param.Integer(default=None, doc="""
        How densely to sample the rasterized plot along the y-axis.""")

    filter_stream = param.ClassSelector(default=FilterStream(), class_=FilterStream, doc="""
        Filter stream to update plot with currently selected filters.""")

    range_stream = param.ClassSelector(default=RangeXY(), class_=RangeXY, doc="""
        Range stream to share between plots to link and persist plot ranges.""")

    flags = param.List(default=[], doc="Flags to select.")

    bad_flags = param.List(default=[], doc="Flags to ignore")

    def __call__(self, dset, **params):
        self.p = ParamOverrides(self, params)

        if self.p.vdim is None:
            vdim = dset.vdims[0].name
        else:
            vdim = self.p.vdim

        ra_range = (ra0, ra1) = dset.range('ra')
        if self.p.ra_sampling:
            xsampling = (ra1-ra0)/self.p.ra_sampling
        else:
            xsampling = None

        dec_range = (dec0, dec1) = dset.range('dec')
        if self.p.dec_sampling:
            ysampling = (dec1-dec0)/self.p.dec_sampling
        else:
            ysampling = None

        if self.p.aggregator == 'mean':
            aggregator = ds.mean(vdim)
        elif self.p.aggregator == 'std':
            aggregator = ds.std(vdim)
        elif self.p.aggregator == 'count':
            aggregator = ds.count()

        sky_range = RangeXY()
        if self.p.range_stream:
            def redim(dset, x_range, y_range):
                ranges = {}
                if x_range and all(isfinite(v) for v in x_range):
                    ranges['ra'] = x_range
                if y_range and all(isfinite(v) for v in x_range):
                    ranges['dec'] = y_range
                return dset.redim.range(**ranges) if ranges else dset
            dset = dset.apply(redim, streams=[self.p.range_stream])
            link_streams(self.p.range_stream, sky_range)
        streams = [sky_range, PlotSize()]

        pts = dset.apply(skypoints, streams=[self.p.filter_stream])

        reset = PlotReset(source=pts)
        reset.add_subscriber(partial(reset_stream, None, [self.p.range_stream]))

        rasterize_inst = rasterize.instance(
            aggregator=aggregator, streams=streams,
            x_sampling=xsampling, y_sampling=ysampling
        )
        raster_pts = apply_when(
            pts, operation=rasterize_inst,
            predicate=lambda pts: len(pts) > self.p.max_points
        )
        return raster_pts.opts(
            opts.Image(bgcolor='black', colorbar=True, cmap=self.p.cmap,
                       min_height=100, responsive=True, tools=['hover'],
                       symmetric=True
            ),
            opts.Points(color=vdim, cmap=self.p.cmap, framewise=True,
                        size=self.p.decimate_size, tools=['hover'],
                        symmetric=True
            ),
            opts.Overlay(hooks=[partial(reset_hook, x_range=ra_range,
                                        y_range=dec_range)])
        )


class skyplot_layout(ParameterizedFunction):
    """Layout of skyplots with linked crosshair
    """
    crosshair = param.Boolean(default=True)

    def __call__(self, skyplots, **params):

        self.p = param.ParamOverrides(self, params)

        pointer = hv.streams.PointerXY(x=0, y=0)
        cross_opts = dict(style={'line_width': 1, 'color': 'black'})
        cross_dmap = hv.DynamicMap(lambda x, y: (hv.VLine(x).opts(**cross_opts) *
                                                 hv.HLine(y).opts(**cross_opts)), streams=[pointer])
        plots = []
        for s in skyplots:
            if self.p.crosshair:
                plot = (s*cross_dmap).relabel(s.label)
            else:
                plot = s
            plots.append(plot)

        return hv.Layout(plots)


class skyshade(Operation):
    """Experimental
    """
    cmap = param.String(default='coolwarm')
    aggregator = param.ObjectSelector(default='mean', objects=['mean', 'std', 'count'])
    width = param.Number(default=None)
    height = param.Number(default=None)
    vdim = param.String(default='y')
    decimate_size = param.Number(default=5)
    max_samples = param.Number(default=10000)

    def _process(self, element, key=None):

        vdim = self.p.vdim
        if self.p.aggregator == 'mean':
            aggregator = ds.mean(vdim)
        elif self.p.aggregator == 'std':
            aggregator = ds.std(vdim)
        elif self.p.aggregator == 'count':
            aggregator = ds.count()

        kwargs = dict(cmap=list(cc.palette[self.p.cmap]),
                      aggregator=aggregator)

        datashaded = dynspread(datashade(element, **kwargs))

        # decimate_opts = dict(plot={'tools':['hover', 'box_select']},
        #                     style={'alpha':0, 'size':self.p.decimate_size,
        #                            'nonselection_alpha':0})
        # decimated = decimate(element, max_samples=self.p.max_samples).opts(**decimate_opts)

        return datashaded.options(responsive=True, height=300)  # * decimated


# # @profile(immediate=True)
# def plot_curve_dask(ddf, kdims=None, vdims=None):
#     import holoviews as hv
#
#     dfc = ddf
#     kdims = kdims or ['visit', 'metrics']
#     vdims = vdims or ['median']
#     dfc = dfc.astype({'visit':str})
#
#     ds = hv.Dataset(dfc, kdims=vdims, vdims=vdims)
#
#     curve = ds.to(hv.Curve, kdims=kdims[0], vdims=vdims[0])#.overlay(kdims[1])
#     points = ds.to(hv.Scatter, kdims=kdims[0], vdims=vdims[0]).opts(size=8, line_color='white')
#
#     plot = (curve * points).opts(hv.opts.Scatter(tools=['hover']))
#
#     curve = curve.redim(y=hv.Dimension(vdims[0], range=(-1, 1)))
#
#     # Now we rename the axis
#     xlabel = kdims[0]
#     ylabel = 'normalized {}'.format(vdims[0])
#
#     grid_style = {'grid_line_color': 'white', 'grid_line_alpha': 0.2}
#     plot = plot.opts(show_legend=False, show_grid=True,
#                      gridstyle=grid_style,
#                      xlabel=xlabel, ylabel=ylabel,
#                      # xticks=100,
#                      responsive=True, aspect=5,
#                      bgcolor='black', xrotation=45)
#     return plot
#
# def _visit_plot(df, metric):
#     def transform_scale(df, metric):
#         with pd.option_context('mode.use_inf_as_na', True):
#             df = df.dropna(subset=[metric])
#             df['result'] = minmax_scale(df[metric])
#         return df
#
#     _meta = df.dtypes.to_dict()
#     _meta.update({'result': float})
#
#     visit_stats = (df
#                     .map_partitions(lambda _df:transform_scale(_df,metric),
#                         meta=_meta)
#                     .groupby('visit')
#                     .result.apply(pd.Series.median, meta=('median',float))
#                     .reset_index().rename(columns={'index':'visit'}))
#
#     return plot_curve_dask(visit_stats)
#
# def _visit_plot_pandas(df, metric):
#     # drop inf/nan values
#     with pd.option_context('mode.use_inf_as_na', True):
#         df = df.dropna(subset=[metric])
#     label = '{!s}'.format(metric)
#     df[label] = minmax_scale(df[metric])
#     df = df.groupby('visit')
#     df = df[label].median().reset_index()
#     df.head().to_csv('bla.csv')
#     return plot_curve_dask(df.rename(columns={metric:'median'}))
#
# # @profile(immediate=True)
# def visits_plot(dsets_visits, filters_to_metrics, filt):
#     # plots = {}
#     metrics = filters_to_metrics[filt]
#     plot_filt = None
#     if metrics:
#     # for filt, metrics in filters_to_metrics.items():
#         # plot_filt = None
#         dset_filt = dsets_visits[filt]
#         for metric in metrics:
#             # plot_metric = _visit_plot(dset_filt[metric], metric)
#             plot_metric = _visit_plot_pandas(dset_filt[metric].compute(), metric)
#             if plot_filt is None:
#                 plot_filt = plot_metric
#             else:
#                 plot_filt = plot_filt * plot_metric
#
#         # if plot_filt is None:
#         #     continue
#         # plots[filt] = plot_filt
#
#     # filters = sorted(plots.keys())
#     # tabs = [(filt, pn.panel(plots[filt])) for filt in filters]
#     # return pn.Tabs(*tabs, sizing_mode='stretch_both')
#     return plot_filt
