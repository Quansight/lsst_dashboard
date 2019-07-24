from functools import partial

import param
import holoviews as hv
import datashader as ds
import colorcet as cc

from param import ParameterizedFunction, ParamOverrides
from bokeh.palettes import Greys9
from holoviews.core.operation import Operation
from holoviews.streams import Stream, BoundsXY, LinkedStream
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.operation.datashader import datashade, dynspread, rasterize
from holoviews.operation import decimate
decimate.max_samples = 5000


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
# All this enables bokeh "reset" button to also reset a stream (such as FilterStream) #
# Not sure if some of this should be updated for newer version of HV, as this was put #
# together circa v1.9.0, I think

class ResetCallback(Callback):
    models = ['plot']
    on_events = ['reset']


class Reset(LinkedStream):
    def __init__(self, *args, **params):
        super(Reset, self).__init__(self, *args, **dict(params, transient=True))


Stream._callbacks['bokeh'][Reset] = ResetCallback

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


def reset_stream(filter_stream):
    filter_stream.event(filter_range={}, flags=[], bad_flags=[])


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
    scatter_cmap = param.String(default='fire', doc="""
        Colormap to use for the scatter plot""")
    sky_cmap = param.String(default='coolwarm', doc="""
        Colormap to use for the sky plot""")
    height = param.Number(default=300, doc="""
        Height in pixels of the combined layout""")
    width = param.Number(default=900, doc="""
        Width in pixels of the combined layout""")
    filter_stream = param.ClassSelector(default=FilterStream(), class_=FilterStream,
                                        doc="Stream to which selection ranges get added.")
    show_rawsky = param.Boolean(default=False, doc="""
        Whether to show the "unselected" sky points in greyscale when there is a selection.""")

    def __call__(self, dset, **params):
        self.p = ParamOverrides(self, params)
        if self.p.xdim not in dset.dimensions():
            raise ValueError('{} not in Dataset.'.format(self.p.xdim))
        if self.p.ydim not in dset.dimensions():
            raise ValueError('{} not in Dataset.'.format(self.p.ydim))
        if ('ra' not in dset.dimensions()) or ('dec' not in dset.dimensions()):
            raise ValueError('ra and/or dec not in Dataset.')

        # Set up scatter plot
        scatter_filterpoints = filterpoints.instance(xdim=self.p.xdim, ydim=self.p.ydim)
        scatter_pts = hv.util.Dynamic(dset, operation=scatter_filterpoints,
                                      streams=[self.p.filter_stream])
        scatter_opts = dict(plot={'height': self.p.height, 'width': self.p.width - self.p.height},
                            norm=dict(axiswise=True))
        scatter_shaded = datashade(scatter_pts, cmap=cc.palette[self.p.scatter_cmap])
        scatter = dynspread(scatter_shaded).opts(**scatter_opts)

        # Set up sky plot
        sky_filterpoints = filterpoints.instance(xdim='ra', ydim='dec', set_title=False)
        sky_pts = hv.util.Dynamic(dset, operation=sky_filterpoints,
                                  streams=[self.p.filter_stream])
        sky_opts = dict(plot={'height': self.p.height, 'width': self.p.height + 100},  # cmap width?
                        norm=dict(axiswise=True))
        sky_shaded = rasterize(sky_pts,
                               aggregator=ds.mean(self.p.ydim), height=self.p.height,
                               width=self.p.width).options(colorbar=True,
                                                           cmap=cc.palette[self.p.sky_cmap])
        sky = sky_shaded.opts(**sky_opts)
        # sky = dynspread(sky_shaded).opts(**sky_opts)

        # Set up summary table
        table = hv.util.Dynamic(dset, operation=summary_table.instance(ydim=self.p.ydim),
                                streams=[self.p.filter_stream])
        table = table.opts(plot={'width': 200})

        # Set up BoundsXY streams to listen to box_select events and notify FilterStream
        scatter_select = BoundsXY(source=scatter)
        scatter_notifier = partial(notify_stream, filter_stream=self.p.filter_stream,
                                   xdim=self.p.xdim, ydim=self.p.ydim)
        scatter_select.add_subscriber(scatter_notifier)

        sky_select = BoundsXY(source=sky)
        sky_notifier = partial(notify_stream, filter_stream=self.p.filter_stream,
                               xdim='ra', ydim='dec')
        sky_select.add_subscriber(sky_notifier)

        # Reset
        reset = Reset(source=scatter)
        reset.add_subscriber(partial(reset_stream, self.p.filter_stream))

        raw_scatter = datashade(scatter_filterpoints(dset), cmap=Greys9[::-1][:5])

        if self.p.show_rawsky:
            raw_sky = datashade(sky_filterpoints(dset), cmap=Greys9[::-1][:5])
            return (table + raw_scatter*scatter + raw_sky*sky)

        else:
            return (table + raw_scatter*scatter + sky)


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
                       for ydim in self._get_ydims(dset)]).cols(3).opts(plot={'merge_tools':False})


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

        return hv.Points(dset, kdims=['ra', 'dec'], vdims=dset.vdims + ['label'])


class skyplot(ParameterizedFunction):
    """Datashaded + decimated RA/dec plot, with colormap of third dimension
    """
    cmap = param.String(default='coolwarm', doc="""
        Colormap to use.""")
    aggregator = param.ObjectSelector(default='mean', objects=['mean', 'std', 'count'], doc="""
        Aggregator for datashading.""")
    vdim = param.String(default=None, doc="""
        Dimension to use for colormap.""")
    width = param.Number(default=None)
    height = param.Number(default=None)
    decimate_size = param.Number(default=5, doc="""
        Size of (invisible) decimated points.""")

    filter_stream = param.ClassSelector(default=FilterStream(), class_=FilterStream)
    flags = param.List(default=[], doc="""
        Flags to select.""")
    bad_flags = param.List(default=[], doc="""
        Flags to ignore""")

    def __call__(self, dset, **params):
        self.p = ParamOverrides(self, params)

        if self.p.vdim is None:
            vdim = dset.vdims[0].name
        else:
            vdim = self.p.vdim

        pts = hv.util.Dynamic(dset, operation=skypoints,
                              streams=[self.p.filter_stream])

        if self.p.aggregator == 'mean':
            aggregator = ds.mean(vdim)
        elif self.p.aggregator == 'std':
            aggregator = ds.std(vdim)
        elif self.p.aggregator == 'count':
            aggregator = ds.count()

        kwargs = dict(cmap=cc.palette[self.p.cmap],
                      aggregator=aggregator)
        if self.p.width is not None:
            kwargs.update(width=self.p.width, height=self.p.height)
#                          streams=[hv.streams.RangeXY])

        decimate_opts = dict(plot={'tools': ['hover', 'box_select']},
                             style={'alpha': 0, 'size': self.p.decimate_size,
                                    'nonselection_alpha': 0})

        decimated = decimate(pts).opts(**decimate_opts)
        sky_shaded = datashade(pts, **kwargs)

        return dynspread(sky_shaded) * decimated


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

        kwargs = dict(cmap=cc.palette[self.p.cmap],
                      aggregator=aggregator)
        if self.p.width is not None:
            kwargs.update(width=self.p.width, height=self.p.height,
                          streams=[hv.streams.RangeXY])

        datashaded = dynspread(datashade(element, **kwargs))

        # decimate_opts = dict(plot={'tools':['hover', 'box_select']},
        #                     style={'alpha':0, 'size':self.p.decimate_size,
        #                            'nonselection_alpha':0})

        # decimated = decimate(element, max_samples=self.p.max_samples).opts(**decimate_opts)

        return datashaded  # * decimated
