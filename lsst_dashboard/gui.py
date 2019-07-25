import logging
from functools import partial

import param
import panel as pn
import holoviews as hv
import numpy as np

from .base import Application
from .base import Component

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


def get_metric_categories():

    categories = ['Photometry', 'Astrometry', 'Shape', 'Color']

    return categories


def get_tract_count():
    return np.random.randint(10e3, 10e4, size=(1))[0]


def get_patch_count():
    return np.random.randint(10e5, 10e7, size=(1))[0]


def get_visit_count():
    return np.random.randint(10e5, 10e7, size=(1))[0]


def get_filter_count():
    return np.random.randint(10e5, 10e7, size=(1))[0]


def get_unique_object_count():
    return np.random.randint(10e5, 10e7, size=(1))[0]


class QuickLookComponent(Component):

    logo = param.String('https://www.lsst.org/sites/default/files/logos/LSST_web_white.png', doc="""
        The logo to display in the header.""")

    data_repository = param.String()

    tract_count = param.Number(default=0)

    patch_count = param.Number(default=0)

    visit_count = param.Number(default=0)

    filter_count = param.Number(default=0)

    unique_object_count = param.Number(default=0)

    comparison = param.String()

    selected = param.Tuple(default=(None, None, None, None), length=4)

    selected_metrics_by_filter = param.Dict(default={'HSC-G': [],
                                                     'HSC-R': [],
                                                     'HSC-I': [],
                                                     'HSC-Y': [],
                                                     'HSC-Z': []})

    label = param.String(default='Quick Look')

    def __init__(self, **param):
        super().__init__(**param)

        self.logo_png = pn.pane.PNG(self.logo,
                                    width=400 // 3,
                                    height=150 // 3)

        text = '<h3><i>Data Processing Explorer</i></h3>'

        self._title = pn.pane.HTML(text,
                                   margin=(0, 0),
                                   height=50)

        self._submit_repository = pn.widgets.Button(
            name='Submit', width=50, align='end')
        self._submit_comparison = pn.widgets.Button(
            name='Submit', width=50, align='end')
        self._submit_repository.on_click(self._update)
        self._submit_comparison.on_click(self._update)
        self._info = pn.pane.HTML(sizing_mode='stretch_width', max_height=10)
        self._metric_panels = []
        self._metric_layout = pn.Column()
        self._plot_layout = pn.Column(sizing_mode='stretch_width')
        self._update(None)

    def title(self):
        return 'Data Processing Explorer'

    def update_selected_by_filter(self, filter_type, selected_values):
        self.selected_metrics_by_filter.update({filter_type: selected_values})
        self.param.trigger('selected_metrics_by_filter')

    def _update(self, event):
        self._update_info()
        self._load_metrics()

    @param.depends('tract_count', 'patch_count', 'visit_count',
                   'filter_count', 'unique_object_count', watch=True)
    def _update_info(self):
        """
        Updates the _info HTML pane with info loaded
        from the current repository.
        """
        html = """
        <b>Tracts:</b> {} |
        <b>Patches:</b> {} |
        <b>Visits:</b> {} |
        <b>Unique Objects:</b> {} |
        <b>Filters (visits):</b> HSC-G (25), HSG-R (24), HSC-I (22),
        HSC-Y (30), HSC-Z (28)
        """.format(format(self.tract_count, ','),
                   format(self.patch_count, ','),
                   format(self.visit_count, ','),
                   format(self.unique_object_count, ','))
        self._info.object = html

    def update_info_counts(self):
        self.tract_count = get_tract_count()
        self.patch_count = get_patch_count()
        self.visit_count = get_visit_count()
        self.unique_object_count = get_unique_object_count()

    def _load_metrics(self):
        """
        Populates the _metrics Row with metrics loaded from the repository
        """
        # Load filters from repository
        filters = ['HSC-G', 'HSG-R', 'HSC-I', 'HSC-Y', 'HSC-Z']

        # TODO: Here there was a performance issue
        # with rendering too many checkboxes
        panels = [MetricPanel(metric='LSST', filters=filters, parent=self)]
        self._metric_panels = panels
        self._metric_layout.objects = [p.panel() for p in panels]

    @param.depends('selected_metrics_by_filter', watch=True)
    def _update_selected_metrics_by_filter(self):

        top_plot = create_top_metric_line_plot('',
                                               self.selected_metrics_by_filter)

        self._plot_layout.clear()
        self._plot_layout.append(top_plot)

        for filt, plots in self.selected_metrics_by_filter.items():
            for p in plots:
                plot = create_metric_star_plot('{} - {}'.format(filt, p))
                self._plot_layout.append(plot)

    def panel(self):
        row1 = pn.Row(self.param.data_repository, self._submit_repository)
        row2 = pn.Row(self.param.comparison, self._submit_comparison)

        return pn.Column(
            pn.Row(self.logo_png, self._title,
                   pn.Spacer(sizing_mode='stretch_width'),
                   pn.Spacer(width=40), row1,
                   pn.Spacer(width=40), row2),
            pn.pane.HTML('<hr width=100%>', sizing_mode='stretch_width',
                         max_height=5),
            pn.Row(self._info),
            pn.pane.HTML('<hr width=100%>', sizing_mode='stretch_width',
                         max_height=10),
            pn.Row(
                self._metric_layout,
                pn.Column(
                    self._plot_layout,
                    sizing_mode='stretch_width'
                )
            ),
            sizing_mode='stretch_both'
        )


class MetricPanel(param.Parameterized):
    """
    A MetricPanel displays checkboxs grouped by filter type group for a particular metric,
    broken down into separate tabs for each filter.
    """

    metric = param.String()

    parent = param.ClassSelector(class_=QuickLookComponent)

    filters = param.List()

    def __init__(self, **params):
        super().__init__(**params)

        self._streams = []
        self._chkbox_groups = [(filt, self._create_metric_checkbox_group(filt))
                               for filt in self.filters]

    def _create_metric_checkbox_group(self, filt):
        metrics = get_available_metrics()
        chkbox_group = MetricCheckboxGroup(metrics)

        chkbox_group.param.watch(partial(self._checkbox_callback, filt),
                                 'metrics')
        widget_kwargs = dict(metrics=pn.widgets.CheckBoxGroup)
        return pn.panel(chkbox_group.param, widgets=widget_kwargs,
                        show_name=False)

    def _checkbox_callback(self, filt, event):
        self.parent.selected = (filt, event.new, filt, event.new)
        self.parent.update_selected_by_filter(filt, event.new)
        self.parent.update_info_counts()

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
dashboard = Application(body=QuickLookComponent())
