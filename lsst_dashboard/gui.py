import param
import panel as pn
import numpy as np
import holoviews as hv

from functools import partial

from .base import Application, Component, TabComponent


class QuickLookComponent(Component):
    
    data_repository = param.String()
    
    comparison = param.String()
    
    selected = param.Tuple(default=(None, None, None, None), length=4)
    
    label = param.String(default='Quick Look')

    def __init__(self, **param):
        super().__init__(**param)
        self._submit_repository = pn.widgets.Button(
            name='Submit', width=50, align='end')
        self._submit_comparison = pn.widgets.Button(
            name='Submit', width=50, align='end')
        self._open_detailed = pn.widgets.Button(
            name='Open Detailed View', disabled=any(v is None for v in self.selected))
        self._submit_repository.on_click(self._update)
        self._submit_comparison.on_click(self._update)
        self._open_detailed.on_click(self._open_detailed_view)
        self._info = pn.pane.HTML(width=600)
        self._metric_panels = []
        self._metric_layout = pn.Column()
        self._plot_layout = pn.Row('A plot will appear here')
        self._update(None)

    def title(self):
        return 'LSST Data Processing Explorer - Quick Look'
    
    def _update(self, event):
        self._update_info()
        self._load_metrics()
    
    def _update_info(self):
        """
        Updates the _info HTML pane with info loaded from the current repository.
        """
        html = """
        <code>
        Tracts: 8, patches = 648<br>
        Visits: 129<br>
        Filters (visits): HSC-G (25), HSG-R (24), HSC-I (22), HSC-Y (30), HSC-Z (28)<br>
        Unique Objects: 8,250,442
        </code>
        """
        self._info.object = html
        
    def _open_detailed_view(self, event):
        """
        This opens the detail dashboard in a new Tab.
        """
        self.parent.current = DetailComponent(
            parent=self.parent, overview=self)

    def _load_metrics(self):
        """
        Populates the _metrics Row with metrics loaded from the repository
        """
        # Load filters from repository
        filters = ['HSC-G', 'HSG-R', 'HSC-I', 'HSC-Y', 'HSC-Z']
        # Load metrics from repository
        metrics = ['Photometry', 'Astrometry', 'Shape', 'Color']
        panels = [MetricPanel(metric=metric, filters=filters, parent=self) for metric in metrics]
        self._metric_panels = panels
        self._metric_layout.objects = [p.panel() for p in panels]
        
    @param.depends('selected', watch=True)
    def _update_selected(self):
        self._plot_layout[0] = mock_plot(self.selected)        
        self._open_detailed.disabled = False

    def panel(self):
        row1 = pn.Row(self.param.data_repository, self._submit_repository)
        row2 = pn.Row(self.param.comparison, self._submit_comparison)
        return pn.Column(
            pn.Row(
                pn.Column(row1, row2),
                pn.layout.HSpacer(),
                self._info
            ),
            pn.pane.HTML('<hr width=100%>', sizing_mode='stretch_width'),
            pn.Row(
                self._metric_layout,
                pn.Column(
                    self._plot_layout,
                    self._open_detailed
                )
            ),
            sizing_mode='stretch_both'
        )


class MetricPanel(param.Parameterized):
    """
    A MetricPanel displays clickable heatmaps for a particular metric,
    broken down into separate tabs for each filter.
    """
    
    metric = param.String()

    parent = param.ClassSelector(class_=QuickLookComponent)
    
    filters = param.List()
    
    def __init__(self, **params):
        super().__init__(**params)
        self._streams = []
        self._heatmaps = [(filt, self._create_metric_heatmap(filt)) for filt in self.filters]

    def _create_metric_heatmap(self, filt):
        """
        Access repository from parent and populate heatmap
        """
        heatmap = hv.HeatMap((range(5), range(2), np.random.randint(0, 2, (2, 5)))).opts(
            height=100, cmap=['red', 'green'], xaxis=None, yaxis=None, toolbar=None,
            line_alpha=1, line_width=5, line_color='white', width=500, nonselection_fill_alpha=0.2)
        tap = hv.streams.Tap(source=heatmap)
        self._streams.append(tap)
        tap.param.watch(partial(self._tapped, filt), ['x', 'y'])
        return heatmap
    
    def _tapped(self, filt, *args):
        """
        This method is called with information about the metric that was clicked.
        """
        self.parent.selected = (self.metric, filt, args[0].obj.x, args[0].obj.y)

    def panel(self):
        return pn.Column(
            pn.pane.Markdown('### %s Metrics' % self.metric, margin=0),
            pn.Tabs(*self._heatmaps, sizing_mode='stretch_width', margin=0),
            sizing_mode='stretch_width'
        )


class DetailComponent(Component):
    """
    This is a mockup of the detailed dashboard.
    """
    
    overview = param.ClassSelector(class_=QuickLookComponent)

    label = param.String(default='Detail')
    
    def title(self):
        return 'LSST Data Processing Explorer - Detailed View'
    
    def panel(self):
        return self.overview._selected_info


def mock_plot(selected):

    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Spectral6
    from bokeh.plotting import figure
    from bokeh.transform import factor_cmap

    fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
    counts = [5, 3, 4, 2, 4, 6]

    source = ColumnDataSource(data=dict(fruits=fruits, counts=counts))

    p = figure(x_range=fruits, plot_height=350, toolbar_location=None, title=str(selected))
    p.vbar(x='fruits', top='counts', width=0.9, source=source, legend="fruits",
           line_color='white', fill_color=factor_cmap('fruits', palette=Spectral6, factors=fruits))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    return p


import holoviews as hv
hv.extension('bokeh')
pn.extension()
dashboard = Application(body=TabComponent(QuickLookComponent()))
