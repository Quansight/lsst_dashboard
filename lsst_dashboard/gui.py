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
# from .plots import visits_plot
from .plots import scattersky, FilterStream, skyplot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('dashboard.log'))


_filters = ['HSC-R', 'HSC-Z', 'HSC-I', 'HSC-G']#, 'HSC-Y']

def init_dataset():
    from .dataset import Dataset
    from .qa_dataset import QADataset
    tables = ['analysisCoaddTable_forced', 'analysisCoaddTable_unforced', 'visitMatchTable']
    tracts = ['9697', '9813', '9615']
    d = Dataset('examples/sample_data')
    d.connect()
    d.load_from_hdf()
    datasets = {}
    for filt in _filters:
        try:
            dtf = d.tables[filt]
            df = dtf['analysisCoaddTable_forced']['9615']
            dataset = QADataset(df)
        except:
            dataset = None
        datasets[filt] = dataset
    datavisits = {}
    for filt in _filters:
        dvf = d.visits[filt]
        dataset_v = dvf['9615']
        datavisits[filt] = dataset_v
    return datasets,datavisits

datasets,datavisits = init_dataset()


def get_available_metrics(filt):
    # metrics = ['base_Footprint_nPix',
    #            'Gaussian-PSF_magDiff_mmag',
    #            'CircAper12pix-PSF_magDiff_mmag',
    #            'Kron-PSF_magDiff_mmag',
    #            'CModel-PSF_magDiff_mmag',
    #            'traceSdss_pixel',
    #            'traceSdss_fwhm_pixel',
    #            'psfTraceSdssDiff_percent',
    #            'e1ResidsSdss_milli',
    #            'e2ResidsSdss_milli',
    #            'deconvMoments',
    #            'compareUnforced_Gaussian_magDiff_mmag',
    #            'compareUnforced_CircAper12pix_magDiff_mmag',
    #            'compareUnforced_Kron_magDiff_mmag',
    #            'compareUnforced_CModel_magDiff_mmag']
    try:
        metrics = datasets[filt].vdims
    except:
        metrics = None
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

    selected_metrics_by_filter = param.Dict(default={f:[] for f in _filters})

    view_mode = ['Skyplot View','Detail View']

    plots_list = []
    skyplot_list = []

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
        self._switch_view = self._create_switch_view_buttons()
        self._plot_top = pn.Row(sizing_mode='stretch_width', margin=(10,10,10,10))
        self._plot_layout = pn.Column(sizing_mode='stretch_width', margin=(10,20,10,20), max_height=500)
        self._update(None)

    def _create_switch_view_buttons(self):
        radio_group = pn.widgets.RadioBoxGroup(name='SwitchView',
                                                options=self.view_mode,
                                                value=self.view_mode[0],
                                                inline=True)
        radio_group.param.watch(self._switch_view_mode, ['value'])
        return radio_group

    def title(self):
        return 'Data Processing Explorer'

    def update_selected_by_filter(self, filter_type, selected_values):
        self.selected_metrics_by_filter.update({filter_type: selected_values})
        self.param.trigger('selected_metrics_by_filter')

    def _update(self, event):
        self._update_info()
        self._load_metrics()

    def create_info_element(self, name, value):

        box_css = """
        background-color: #EEEEEE;
        border: 1px solid #777777;
        display: inline-block;
        padding-left: 5px;
        padding-right: 5px;
        margin-left:7px;
        """

        fval = format(value, ',')
        return '<span style="{}"><b>{}</b> {}</span>'.format(box_css,
                                                             fval,
                                                             name)

    @param.depends('tract_count', 'patch_count', 'visit_count',
                   'filter_count', 'unique_object_count', watch=True)
    def _update_info(self):
        """
        Updates the _info HTML pane with info loaded
        from the current repository.
        """
        html = ''
        html += self.create_info_element('Tracts', self.tract_count)
        html += self.create_info_element('Patches', self.patch_count)
        html += self.create_info_element('Visits', self.visit_count)
        html += self.create_info_element('Unique Objects',
                                         self.unique_object_count)
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
        # filters = ['HSC-G', 'HSG-R', 'HSC-I', 'HSC-Y', 'HSC-Z']
        filters = list(self.selected_metrics_by_filter.keys())

        panels = [MetricPanel(metric='LSST', filters=filters, parent=self)]
        self._metric_panels = panels
        self._metric_layout.objects = [p.panel() for p in panels]

    @param.depends('selected_metrics_by_filter', watch=True)
    def _update_selected_metrics_by_filter(self):

        plots_list = []
        top_plot = create_top_metric_line_plot('',
                                               self.selected_metrics_by_filter)
        # top_plot = visits_plot(datavisits, self.selected_metrics_by_filter)
        plots_list.append(('a',top_plot))
        # self._plot_top.clear()
        # self._plot_top.append(top_plot)

        skyplot_list = []
        for filt, plots in self.selected_metrics_by_filter.items():
            filter_stream = FilterStream()
            dset = datasets[filt]
            for i,p in enumerate(plots):
                # skyplots
                plot_sky = skyplot(dset.ds,
                                    filter_stream=filter_stream,
                                    vdim=p)
                skyplot_list.append((p,plot_sky))

                # plot = create_metric_star_plot('{} - {}'.format(filt, p))
                plots_ss = scattersky(dset.ds,#.groupby('label'),
                                      xdim='psfMag',
                                      ydim=p)
                plot = plots_ss
                # p_sky = plots_ss[2]
                plots_list.append((p,plot))
        self.skyplot_list = skyplot_list
        self.plots_list = plots_list
        self._switch_view_mode()

    def _switch_view_mode(self, *events):
        view_mode = self._switch_view.value
        logging.info(view_mode)
        if len(self.plots_list):
            if view_mode == 'Skyplot View':
                # tab_layout = pn.Tabs(sizing_mode='stretch_width')
                # for i,p in enumerate(self.plots_list):
                #     tab_layout.append((str(i),p))
                tab_layout = pn.Tabs(*self.skyplot_list, sizing_mode='stretch_width')
                # self._plot_layout.clear()
                try:
                    _ = self._plot_layout.pop(0)
                except:
                    pass
                self._plot_layout.css_classes = []
                self._plot_layout.append(tab_layout)
            else:
                list_layout = pn.Column(sizing_mode='stretch_width')
                for i,p in self.plots_list:
                    list_layout.append(p)
                # self._plot_layout.clear()
                try:
                    _ = self._plot_layout.pop(0)
                except:
                    pass
                self._plot_layout.css_classes = ['scrolling_list']
                self._plot_layout.append(list_layout)

    def jinja(self):
        from ._jinja2_templates import quicklook
        import holoviews as hv
        tmpl = pn.Template(quicklook)
        components = [
            ('brand',pn.Row(self.logo_png, self._title)),
            ('row1',pn.Row(self.param.data_repository, self._submit_repository)),
            ('row2',pn.Row(self.param.comparison, self._submit_comparison)),
            ('info',self._info),
            ('metrics_selectors',self._metric_layout),
            ('view_switchers',self._switch_view),
            ('plot_top',self._plot_top),
            ('metrics_plots',self._plot_layout)
        ]
        for l,c in components:
            tmpl.add_panel(l,c)
        return tmpl

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
                    self._switch_view,
                    self._plot_top,
                    self._plot_layout,
                    # self._layout_metrics_plots(self._switch_view.value),
                    sizing_mode='stretch_width',
                ),
            ),
            sizing_mode='stretch_both',
        )

class MetricPanel(param.Parameterized):
    """
    A MetricPanel displays checkboxs grouped by
    filter type group for a particular metric,
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
        metrics = get_available_metrics(filt)
        if not metrics:
            return pn.pane.Markdown("_No metrics available_")
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
_css = '''
.scrolling_list {
    background: #f0f0f0;
    overflow-y: auto;
}
'''
pn.extension(raw_css=[_css])
dashboard = Application(body=QuickLookComponent())
