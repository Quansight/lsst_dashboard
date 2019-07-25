import logging
from functools import partial

import param
import panel as pn
import holoviews as hv

from .base import Application
from .base import Component
from .base import TabComponent

from .plots import create_top_metric_line_plot
from .plots import create_metric_star_plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('dashboard.log'))


def get_available_metrics():

    metrics = ['base_Footprint_nPix',
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
               'compareUnforced_CModel_magDiff_mmag']

    return metrics


class QuickLookComponent(Component):

    data_repository = param.String()

    comparison = param.String()

    selected = param.Tuple(default=(None, None, None, None), length=4)

    selected = param.Tuple(default=(None, None, None, None), length=4)

    selected_metrics_by_filter = param.Dict(default={'HSC-G': [],
                                                     'HSC-R': [],
                                                     'HSC-I': [],
                                                     'HSC-Y': [],
                                                     'HSC-Z': []})

    label = param.String(default='Quick Look')

    def __init__(self, **param):
        super().__init__(**param)
        self._submit_repository = pn.widgets.Button(
            name='Submit', width=50, align='end')
        self._submit_comparison = pn.widgets.Button(
            name='Submit', width=50, align='end')
        self._submit_repository.on_click(self._update)
        self._submit_comparison.on_click(self._update)
        self._info = pn.pane.HTML(width=600)
        self._metric_panels = []
        self._metric_layout = pn.Column()
        self._plot_layout = pn.Column()
        self._update(None)

    def title(self):
        return 'LSST Data Processing Explorer - Quick Look'

    def update_selected_by_filter(self, filter_type, selected_values):
        logger.info('.update_selected_by_filter')
        self.selected_metrics_by_filter.update({filter_type: selected_values})
        self.param.trigger('selected_metrics_by_filter')

        #self._update_selected_metrics_by_filter()

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
        
    def _load_metrics(self):
        """
        Populates the _metrics Row with metrics loaded from the repository
        """
        # Load filters from repository
        filters = ['HSC-G', 'HSG-R', 'HSC-I', 'HSC-Y', 'HSC-Z']

        # Load metrics from repository
        metrics = ['Photometry', 'Astrometry', 'Shape', 'Color']

        # TODO: Here there was a performance issue with rendering too many checkboxes
        panels = [MetricPanel(metric='LSST', filters=filters, parent=self)]
        #panels = [MetricPanel(metric=metric, filters=filters, parent=self) for metric in metrics]
        self._metric_panels = panels
        self._metric_layout.objects = [p.panel() for p in panels]
        
    @param.depends('selected_metrics_by_filter', watch=True)
    def _update_selected_metrics_by_filter(self):
        for filt, plots in self.selected_metrics_by_filter.items():
            for p in plots:
                plot = create_metric_star_plot('{} - {}'.format(filt, p))
                self._plot_layout.append(plot)

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
        self._chkbox_groups = [(filt, self._create_metric_checkbox_group(filt)) for filt in self.filters]

    def _create_metric_checkbox_group(self, filt):
        """
        Access repository from parent and populate heatmap
        """
        metrics = get_available_metrics()
        chkbox_group = MetricCheckboxGroup(metrics)

        chkbox_group.param.watch(partial(self._checkbox_callback, filt), 'metrics')
        widget_kwargs = dict(metrics=pn.widgets.CheckBoxGroup)
        return pn.panel(chkbox_group.param, widgets=widget_kwargs, show_name=False)
        # pn.pane.Param

    def _checkbox_callback(self, filt, event):
        logger.info('._checkbox_callback')
        self.parent.selected = (filt, event.new, filt, event.new)
        self.parent.update_selected_by_filter(filt, event.new)


    def _tapped(self, filt, *args):
        """
        This method is called with information about the metric that was clicked.
        """
        self.parent.selected = (self.metric, filt, args[0].obj.x, args[0].obj.y)

    def panel(self):
        return pn.Column(
            pn.pane.Markdown('### %s Metrics' % self.metric, margin=0),
            pn.Tabs(*self._chkbox_groups, sizing_mode='stretch_width',
                    margin=0),
            sizing_mode='stretch_width'
        )


class MetricCheckboxGroup(param.Parameterized):

    metrics = param.ListSelector(default=[])

    def __init__(self, available_metrics, **kwargs):
        self.param.metrics.objects = available_metrics
        super().__init__(**kwargs)


hv.extension('bokeh')
pn.extension()
dashboard = Application(body=TabComponent(QuickLookComponent()))
