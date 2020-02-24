import traceback
import json
import logging
import os

from collections import defaultdict
from functools import partial

import param
import panel as pn
import holoviews as hv
import numpy as np
import pandas as pd
import sklearn.preprocessing

from holoviews.plotting.bokeh.element import ElementPlot

from .base import Application
from .base import Component

from .plots import visits_plot
from .plots import scattersky, FilterStream, skyplot

from .dataset import Dataset
from .qa_dataset import QADataset

from .utils import set_timeout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('dashboard.log'))

current_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.split(current_directory)[0]

with open(os.path.join(current_directory, 'dashboard.html')) as template_file:
    dashboard_html_template = template_file.read()

pn.extension()

datasets = []
filtered_datasets = []
datavisits = []
filtered_datavisits = []

sample_data_directory = 'sample_data'

class Store(object):

    def __init__(self):
        self.active_dataset = Dataset('')


def init_dataset(data_repo_path, datastack='qaDashboardCoaddTable', **kwargs):

    global datasets
    global filtered_datasets
    global datavisits
    global filtered_datavisits

    d = Dataset(data_repo_path, **kwargs)
    d.connect()

    global store
    store.active_dataset = d

    flags = d.flags

    datasets = {}
    filtered_datasets = {}
    dtf = d.coadd[datastack]

    dtf = dtf.set_index('filter')
    for filt in d.filters:
        df = dtf.loc[filt]
        datasets[filt] = QADataset(df)
        filtered_datasets[filt] = QADataset(df.copy())

    datavisits = {}
    filtered_datavisits = {}
    for filt in d.filters:
        datavisits[filt] = {}
        filtered_datavisits[filt] = {}
        for metric in d.metrics:
            df = d.visits_by_metric[filt][metric]
            filtered_df = None
            if df is not None:
                df = df
                filtered_df = df.copy()

            datavisits[filt][metric] = df
            filtered_datavisits[filt][metric] = filtered_df

    return d


# TODO: build and cache summarized visits dataframe
def summarize_visits_dataframe(data_repo_path):

    d = Dataset(data_repo_path)
    d.connect()

    dfs = []
    for filt in d.filters:
        for metric in d.metrics:
            if metric in datavisits[filt].columns:
                df = datavisits[filt][metric].reset_index(-1)
                df = pd.DataFrame(getattr(sklearn.preprocessing,
                                          'minmax_scale',
                                          lambda x: x)(df),
                                  index=df.index,
                                  columns=df.columns).groupby(df.index)

                values = df[metric].median().values
                df = pd.DataFrame(dict(median_norm=values, filter_type=filt,
                                       metric=metric))
                dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_data(data_repo_path=None, datastack = 'forced'):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.split(current_directory)[0]

    if not data_repo_path:
        data_repo_path = sample_data_directory

    if not os.path.exists(data_repo_path):
        raise ValueError('Data Repo Path does not exist.')

    datastack = 'qaDashboardCoaddTable' # + datastack -- disabled forced/unforced for now
    d = init_dataset(data_repo_path, datastack=datastack)

    return d


store = Store()
#store.active_dataset = load_data()


def get_available_metrics(filt):

    global store
    if not store.active_dataset:
        return None

    return store.active_dataset.metrics


def get_metric_categories():
    categories = ['Photometry', 'Astrometry', 'Shape', 'Color']
    return categories


def get_unique_object_count():
    return np.random.randint(10e5, 10e7, size=(1))[0]


class QuickLookComponent(Component):

    data_repository = param.String(default=sample_data_directory,
                                   label=None, allow_None=True)

    query_filter = param.String(label="Query Expression")

    query_filter_active = param.String(label="Active Query Filter", default='')

    new_column_expr = param.String(label="Data Column Expression")

    tract_count = param.Number(default=0)

    status_message_queue = param.List(default=[])

    patch_count = param.Number(default=0)

    visit_count = param.Number(default=0)

    filter_count = param.Number(default=0)

    unique_object_count = param.Number(default=0)

    comparison = param.String()

    selected = param.Tuple(default=(None, None, None, None), length=4)

    selected_metrics_by_filter = param.Dict(default={f: [] for f in store.active_dataset.filters})

    selected_flag_filters = param.Dict(default={})

    view_mode = ['Skyplot View', 'Detail View']
    data_stack = ['Forced Coadd', 'Unforced Coadd']

    plot_top = None
    plots_list = []
    skyplot_list = []

    label = param.String(default='Quick Look')

    def __init__(self, store, **param):

        super().__init__(**param)

        self.store = store

        self._clear_metrics_button = pn.widgets.Button(
            name='Clear', width=30, align='end')
        self._clear_metrics_button.on_click(self._on_clear_metrics)

        self._submit_repository = pn.widgets.Button(
            name='Load Data', width=50, align='end')
        self._submit_repository.on_click(self._on_load_data_repository)

        self._submit_comparison = pn.widgets.Button(
            name='Submit', width=50, align='end')
        self._submit_comparison.on_click(self._update)

        self.flag_filter_select = pn.widgets.Select(
            name='Add Flag Filter', width=160, options=self.store.active_dataset.flags)

        self.flag_state_select = pn.widgets.Select(
            name='Flag State', width=75, options=['True', 'False'])

        self.flag_submit = pn.widgets.Button(
            name='Add Flag Filter', width=10, height=30, align='end')
        self.flag_submit.on_click(self.on_flag_submit_click)

        self.flag_filter_selected = pn.widgets.Select(
            name='Active Flag Filters', width=250)

        self.flag_remove = pn.widgets.Button(
            name='Remove Flag Filter', width=50, height=30, align='end')
        self.flag_remove.on_click(self.on_flag_remove_click)

        self.query_filter_submit = pn.widgets.Button(
            name='Run Query Filter', width=100, align='end')
        self.query_filter_submit.on_click(self.on_run_query_filter_click)

        self.query_filter_clear = pn.widgets.Button(
            name='Clear', width=50, align='end')
        self.query_filter_clear.on_click(self.on_query_filter_clear)

        self.new_column_submit = pn.widgets.Button(
            name='Define New Column', width=100, align='end')
        self.new_column_submit.on_click(self.on_define_new_column_click)

        self.status_message = pn.pane.HTML(sizing_mode='stretch_width', max_height=10)
        self.adhoc_js = pn.pane.HTML(sizing_mode='stretch_width', max_height=10)
        self._info = pn.pane.HTML(sizing_mode='stretch_width', max_height=10)
        self._flags = pn.pane.HTML(sizing_mode='stretch_width', max_height=10)
        self._metric_panels = []
        self._metric_layout = pn.Column()
        self._switch_view = self._create_switch_view_buttons()
        self._switch_stack = self._create_switch_datastack_buttons()
        self._plot_top = pn.Row(sizing_mode='stretch_width',
                                margin=(10, 10, 10, 10))

        self._plot_layout = pn.Column(sizing_mode='stretch_width',
                                      margin=(10, 10, 10, 10))

        self.skyplot_layout = pn.Column(sizing_mode='stretch_width',
                                      margin=(10, 10, 10, 10))

        self.list_layout = pn.Column(sizing_mode='stretch_width')

        self._update(None)

    def _on_load_data_repository(self, event):

        global datasets
        global datavisits
        global filtered_datavisits

        self.store.active_dataset = Dataset('')
        self.skyplot_list = []
        self.plots_list = []
        self.plot_top = None

        datasets = {}
        filtered_datasets = {}
        datavisits = {}
        filtered_datavisits = {}

        self._load_metrics()
        self._switch_view_mode()
        self.update_display()

        data_repo_path = self.data_repository
        self.add_status_message('Load Data Start...', data_repo_path,
                                level='info')

        dstack_switch_val = self._switch_stack.value.lower()
        datastack = 'unforced' if 'unforced' in dstack_switch_val else 'forced'
        try:
            self.store.active_dataset = load_data(data_repo_path, datastack)

        except Exception as e:
            self.update_display()
            self.add_message_from_error('Data Loading Error',
                                        data_repo_path, e)
            raise

        self.add_status_message('Data Ready', data_repo_path,
                                level='success', duration=3)
        # update ui
        self.flag_filter_select.options = self.store.active_dataset.flags

        for f in self.store.active_dataset.filters:
            self.selected_metrics_by_filter[f] = []

        self._load_metrics()
        self._switch_view_mode()
        self.update_display()

    def update_display(self):
        self.set_checkbox_style()

    def set_checkbox_style(self):
        code = '''$("input[type='checkbox']").addClass("metric-checkbox");'''
        self.execute_js_script(code)

        global store
        for filter_type, fails in store.active_dataset.failures.items():
            error_metrics = json.dumps(fails)
            code = '$(".' + filter_type + '-checkboxes .metric-checkbox").siblings().filter(function () { return ' + error_metrics + '.indexOf($(this).text()) > -1;}).css("color", "orange");'
            self.execute_js_script(code)

    def add_status_message(self, title, body, level='info', duration=5):
        msg = {'title': title, 'body': body}
        msg_args = dict(msg=msg, level=level, duration=duration)
        self.status_message_queue.append(msg_args)
        self.param.trigger('status_message_queue')  # to work with panel 0.7
        # Drop message in terminal/logger too
        try:
            # temporary try/except until 'level' values are all checked
            getattr(logger,level)(msg)
        except:
            pass

    def on_flag_submit_click(self, event):
        flag_name = self.flag_filter_select.value
        flag_state = self.flag_state_select.value == 'True'
        self.selected_flag_filters.update({flag_name: flag_state})
        self.param.trigger('selected_flag_filters')
        self.add_status_message('Added Flag Filter',
                                '{} : {}'.format(flag_name, flag_state),
                                level='info')

    def on_flag_remove_click(self, event):
        flag_name = self.flag_filter_selected.value.split()[0]
        del self.selected_flag_filters[flag_name]
        self.param.trigger('selected_flag_filters')
        self.add_status_message('Removed Flag Filter',
                                flag_name, level='info')

    def on_run_query_filter_click(self, event):
        self.query_filter_active = self.query_filter
        self.query_filter = ''

    def on_query_filter_clear(self, event):
        self.query_filter_active = ''

    def _on_clear_metrics(self, event):
        for k in self.selected_metrics_by_filter.keys():
            self.selected_metrics_by_filter[k] = []
        self.param.trigger('selected_metrics_by_filter')
        code = '''$("input[type='checkbox']").prop("checked", false);'''
        self.execute_js_script(code)

    def on_define_new_column_click(self, event):
        new_column_expr = self.new_column_expr
        logger.info("NEW COLUMN EXPRESSION: '{!s}'".format(new_column_expr))

    def _create_switch_view_buttons(self):
        radio_group = pn.widgets.RadioBoxGroup(name='SwitchView',
                                               options=self.view_mode,
                                               align='center',
                                               value=self.view_mode[0],
                                               inline=True)
        radio_group.param.watch(self._switch_view_mode, ['value'])
        return radio_group

    def _create_switch_datastack_buttons(self):
        radio_group = pn.widgets.RadioBoxGroup(name='SwitchDataStack',
                                               options=self.data_stack,
                                               align='center',
                                               value=self.data_stack[0],
                                               inline=True)
        radio_group.param.watch(self._switch_data_stack, ['value'])
        return radio_group

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
        outel = '<li><span style="{}"><b>{}</b> {}</span></li>'
        return outel.format(box_css,fval,name)

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
        # html += self.create_info_element('Unique Objects',
        #                                  self.unique_object_count)
        self._info.object = '<ul class="list-group list-group-horizontal" style="list-style: none;">{}</ul>'.format(html)

    def create_status_message(self, msg, level='info', duration=5):
        import uuid
        msg_id = str(uuid.uuid1())
        color_levels = dict(info='rgba(0,191,255, .8)',
                            error='rgba(249, 180, 45, .8)',
                            warning='rgba(240, 255, 0, .8)',
                            success='rgba(3, 201, 169, .8)')

        box_css = """
        width: 15rem;
        background-color: {};
        border: 1px solid #CCCCCC;
        display: inline-block;
        color: white;
        padding: 5px;
        margin-top: 1rem;
        """.format(color_levels.get(level, 'rgba(0,0,0,0)'))

        remove_msg_func = ('<script>(function() { '
                           'setTimeout(function(){ document.getElementById("'+ msg_id +'").outerHTML = ""; }, ' + str(duration * 1000) +')})()'
                           '</script>')

        text = '<span style="{}"><h5>{}</h5><hr/><p>{}</p></span></li>'.format(box_css, msg.get('title'), msg.get('body') )

        return ('<li id="{}" class="status-message nav-item">'
                '{}'
                '{}'
                '</lil>').format(msg_id, remove_msg_func, text)

    def gen_clear_func(self, msg):
        async def clear_message():
            try:
                if msg in self.status_message_queue:
                    self.status_message_queue.remove(msg)
            except ValueError:
                pass
        return clear_message

    @param.depends('status_message_queue', watch=True)
    def _update_status_message(self):

        queue_css = """
        list-style-type: none;
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        background-color: rgba(0,0,0,0);
        border: none;
        display: inline-block;
        margin-left: 7px;
        """

        html = ''

        for msg in self.status_message_queue:
            html += self.create_status_message(**msg)
            set_timeout(msg.get('duration', 5), self.gen_clear_func(msg))

        self.status_message.object = '<ul style="{}">{}</ul>'.format(queue_css, html)

    def execute_js_script(self, js_body):
        script = '<script>(function() { ' + js_body +  '})()</script>'  # to work with panel 0.7
        self.adhoc_js.object = script

    def get_patch_count(self):
        return 1
        patchs = set()
        for filt,_ in self.selected_metrics_by_filter.items():
            dset = self.get_dataset_by_filter(filt)
            patchs = patchs.union(set(dset.df['patch'].unique()))
        return len(patchs)

    def get_tract_count(self):
        return 1
        tracts = set()
        for filt,_ in self.selected_metrics_by_filter.items():
            dset = self.get_dataset_by_filter(filt)
            tracts = tracts.union(set(dset.df['tract'].unique()))
        return len(tracts)

    def get_visit_count(self):
        return 1
        dvisits = self.get_datavisits()
        visits = set()
        for filt,metrics in self.selected_metrics_by_filter.items():
            for metric in metrics:
                df = dvisits[filt][metric].compute()
                visits = visits.union(set(df['visit'].unique()))
        return len(visits)

    def update_info_counts(self):
        self.tract_count = self.get_tract_count()
        self.patch_count = self.get_patch_count()
        self.visit_count = self.get_visit_count()
        self.unique_object_count = get_unique_object_count()

    def _load_metrics(self):
        """
        Populates the _metrics Row with metrics loaded from the repository
        """
        panels = [MetricPanel(metric='LSST',
                  filters=self.store.active_dataset.filters,
                  parent=self)]
        self._metric_panels = panels

        self._metric_layout.objects = [p.panel() for p in panels]
        self.update_display()

    @param.depends('query_filter_active', watch=True)
    def _update_query_filter(self):
        self.filter_main_dataframe()

    @param.depends('selected_flag_filters', watch=True)
    def _update_selected_flags(self):
        selected_flags = ['{} : {}'.format(f,v)
                          for f,v in self.selected_flag_filters.items()]
        self.flag_filter_selected.options = selected_flags
        self.filter_main_dataframe()

    def filter_main_dataframe(self):
        global filtered_datasets
        global datasets
        for filt, qa_dataset in datasets.items():
            try:
                query_expr = self._assemble_query_expression()
                if query_expr:
                    filtered_datasets[filt] = QADataset(datasets[filt].df.query(query_expr))
            except Exception as e:
                self.add_message_from_error('Filtering Error', '', e)
                raise
                return
        #self.filter_visits_dataframe()
        self._update_selected_metrics_by_filter()

    def filter_visits_dataframe(self):
        global filtered_datavisits
        global datavisits
        for filt, metrics in datavisits.items():
            for metric, df in metrics.items():
                try:
                    query_expr = self._assemble_query_expression(ignore_query_expr=True)
                    if query_expr and datavisits[filt][metric] is not None:
                        filtered_datavisits[filt][metric] = datavisits[filt][metric].query(query_expr)
                except Exception as e:
                    self.add_message_from_error('Filtering Visits Error', '', e)
                    return

    def _assemble_query_expression(self, ignore_query_expr=False):
        query_expr = ''

        flags_query = []
        for flag, state in self.selected_flag_filters.items():
            flags_query.append('{}=={}'.format(flag, state))
        if flags_query:
            query_expr += ' & '.join(flags_query)

        if ignore_query_expr:
            return query_expr

        query_filter = self.query_filter.strip()
        if query_filter:
            if query_expr:
                query_expr += ' & {!s}'.format(query_filter)
            else:
                query_expr = '{!s}'.format(query_filter)

        return query_expr

    def get_dataset_by_filter(self, filter_type):
        global datasets
        global filtered_datasets
        if self.query_filter == '' and len(self.selected_flag_filters) == 0:
            return datasets[filter_type]
        else:
            return filtered_datasets[filter_type]

    def get_datavisits(self):
        global datavisits
        global filtered_datavisits
        # if self.query_filter == '' and len(self.selected_flag_filters) == 0:
        if len(self.selected_flag_filters) == 0:
            return datavisits
        else:
            return filtered_datavisits

    def add_message_from_error(self, title, info, exception_obj, level='error'):

        tb = traceback.format_exception_only(type(exception_obj),
                                             exception_obj)[0]
        msg_body = '<b>Info:</b> ' + info + '<br />'
        msg_body += '<b>Cause:</b> ' + tb
        logger.error(title)
        logger.error(msg_body)
        self.add_status_message(title,
                                msg_body, level=level, duration=10)

    @param.depends('selected_metrics_by_filter', watch=True)
    def _update_selected_metrics_by_filter(self):

        plots_list = []
        skyplot_list = []

        top_plot = None

        dvisits = self.get_datavisits()
        try:
            top_plot = visits_plot(dvisits, self.selected_metrics_by_filter)
        except Exception as e:
            self.add_message_from_error('Visits Plot Error',
                                        '', e)

        self.plot_top = top_plot

        filter_stream_scatter = FilterStream()
        for filt, plots in self.selected_metrics_by_filter.items():
            filter_stream = FilterStream()
            dset = self.get_dataset_by_filter(filt)
            for i, p in enumerate(plots):
                # skyplots
                plot_sky = skyplot(dset.ds,
                                   filter_stream=filter_stream,
                                   vdim=p)

                skyplot_list.append((filt + ' - ' + p, plot_sky))

                plots_ss = scattersky(dset.ds,
                                      xdim='psfMag',
                                      ydim=p,
                                      filter_stream=filter_stream_scatter)
                plot = plots_ss
                plots_list.append((p, plot))

        self.skyplot_list = skyplot_list
        self.plots_list = plots_list

        self.update_display()
        self._switch_view_mode()

    def linked_tab_plots(self):
        tabs = [(name, pn.panel(plot)) for name, plot in self.skyplot_list]
        return pn.Tabs(*tabs, sizing_mode='stretch_both')

    def attempt_to_clear(self, obj):
        try:
            obj.clear()
        except:
            pass

    def _switch_data_stack(self, *events):
        # clear existing plot layouts
        self.attempt_to_clear(self._plot_top)
        self.attempt_to_clear(self._plot_layout)
        self.attempt_to_clear(self.skyplot_layout)
        self.attempt_to_clear(self.list_layout)

        self._on_clear_metrics(event=None)
        self._on_load_data_repository(None)

    def _switch_view_mode(self, *events):
        # clear existing plot layouts
        self.attempt_to_clear(self._plot_top)
        self.attempt_to_clear(self._plot_layout)
        self.attempt_to_clear(self.skyplot_layout)
        self.attempt_to_clear(self.list_layout)

        if self._switch_view.value == 'Skyplot View':
            self.execute_js_script('''$( ".skyplot-plot-area" ).show(); $( ".metrics-plot-area" ).hide();''')
            self.skyplot_layout.append(self.linked_tab_plots())

        else:
            self.execute_js_script('''$( ".skyplot-plot-area" ).hide(); $( ".metrics-plot-area" ).show();''')
            logger.info(self.plot_top)
            self._plot_top.append(self.plot_top)
            for i, p in self.plots_list:
                self.list_layout.append(p)
            self._plot_layout.append(self.list_layout)

    def jinja(self):
        from ._jinja2_templates import quicklook
        import holoviews as hv
        tmpl = pn.Template(dashboard_html_template)

        data_repo_widget = pn.panel(self.param.data_repository,
                                    show_labels=False)
        data_repo_widget.width = 300
#        data_repo_row = pn.Row(pn.panel('Data Repository', align='end'),
#                               data_repo_widget, self._submit_repository)
        data_repo_row = pn.Row(data_repo_widget, self._submit_repository)
        data_repo_row.css_classes = ['data-repo-input']

        query_filter_widget = pn.panel(self.param.query_filter)
        query_filter_widget.width = 260

        query_filter_active_widget = pn.panel(self.param.query_filter_active)
        query_filter_active_widget.width = 260

        new_column_widget = pn.panel(self.param.new_column_expr)
        new_column_widget.width = 260

        datastack_switcher = pn.Row(self._switch_stack)
        datastack_switcher.css_classes = ['stack-switcher']

        view_switcher = pn.Row(self._switch_view)
        view_switcher.css_classes = ['view-switcher']

        clear_button_row = pn.Row(self._clear_metrics_button)

        components = [
            ('metrics_clear_button', clear_button_row),
            ('data_repo_path', data_repo_row),
            ('status_message_queue', self.status_message),
            ('adhoc_js', self.adhoc_js),

            ('infobar', self._info),
#            ('view_switcher', switcher_row),
            ('stack_switcher', datastack_switcher),
            ('view_switcher', view_switcher),

            ('metrics_selectors', self._metric_layout),
            ('metrics_plots', self._plot_layout),
            ('skyplot_metrics_plots', self.skyplot_layout),
            ('plot_top', self._plot_top),
            ('flags', pn.Column(pn.Row(self.flag_filter_select,
                                       self.flag_state_select),
                                pn.Row(self.flag_submit),
                                pn.Row(self.flag_filter_selected),
                                pn.Row(self.flag_remove))),
            ('query_filter', pn.Column(query_filter_widget,
                                       pn.Row(self.query_filter_submit),
                                       query_filter_active_widget,
                                       pn.Row(self.query_filter_clear), css_classes=['readonly-string'])),
            ('new_column', pn.Column(new_column_widget,
                                     pn.Row(self.new_column_submit)),),
        ]

        for l, c in components:
            tmpl.add_panel(l, c)

        return tmpl


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

        global store
        metrics = store.active_dataset.metrics

        if not metrics:
            return pn.pane.Markdown("_No metrics available_")

        chkbox_group = MetricCheckboxGroup(metrics)
        chkbox_group.param.watch(partial(self._checkbox_callback, filt),
                                 'metrics')
        widget_kwargs = dict(metrics=pn.widgets.CheckBoxGroup)
        widg = pn.panel(chkbox_group.param, widgets=widget_kwargs,
                        show_name=False)
        widg.css_classes = [filt + '-checkboxes']
        return widg

    def _checkbox_callback(self, filt, event):
        self.parent.selected = (filt, event.new, filt, event.new)
        self.parent.update_selected_by_filter(filt, event.new)
        self.parent.update_info_counts()
        self.parent.update_display()

    def panel(self):
        return pn.Column(
            pn.Tabs(*self._chkbox_groups, sizing_mode='stretch_width',
                    margin=0),
            sizing_mode='stretch_width'
        )


class MetricCheckboxGroup(param.Parameterized):

    metrics = param.ListSelector(default=[])

    def __init__(self, available_metrics, **kwargs):
        self.param.metrics.objects = sorted(available_metrics)
        super().__init__(**kwargs)


hv.extension('bokeh')
_css = '''
.scrolling_list {
    overflow-y: auto !important;
}

.skyplot-extras {
    height: 100% !important;
    width: 100% !important;
    overflow-y: auto;
}
'''


dashboard = Application(body=QuickLookComponent(store))
