import pandas as pd
import numpy as np
import holoviews as hv


def visits_plot(dsets_visits, filters_to_metrics, filt, errors=[]):
    metrics = filters_to_metrics[filt]
    dset_filt = dsets_visits[filt]
    plot_filt = visits_plot_per_filter(dset_filt, metrics, filt, errors=errors)
    return plot_filt


def visits_plot_layout(plots):
    import panel as pn
    tabs = [(filt, pn.panel(plot)) for filt,plot in plots.items()]
    return pn.Tabs(*tabs, sizing_mode='stretch_both')


def visits_plot_per_filter(dsets_filter, metrics, filt, errors=[]):
    '''
    * dsets_filter: a dictionary pointing to metrics dataframes
    '''
    plot_all = None
    for metric in metrics:
        try:
            df = dsets_filter[metric]
            dfp = process_metric_visits(df.compute(), metric)
            col_median = '{}_median'.format(metric)
            col_scaled = '{}_scaled'.format(metric)
            dfp.rename(columns={'cmedian': col_median,
                                'cscaled': col_scaled},
                       inplace=True)
            plot_metric = visits_plot_per_metric(dfp, 'visit',  col_scaled,
                                                 [col_median, 'visit'],
                                                 filt=filt)
            plot_all = plot_all * plot_metric if plot_all else plot_metric
        except:
            errors.append(metric)

    if plot_all:
        return plot_all.opts(show_legend=False, show_grid=True,
                             gridstyle={'grid_line_color': 'white',
                                        'grid_line_alpha': 0.2},
                             responsive=True, aspect=5,
                             ylabel="normalized/metric",
                             bgcolor='black', xrotation=45)
    else:
        return None


def process_metric_visits(dfc, metric):
    from sklearn.preprocessing import minmax_scale

    with pd.option_context('mode.use_inf_as_na', True):
        dfc = dfc.dropna(subset=[metric])

    dfc['minmax'] = dfc[metric].transform(minmax_scale)
    dfgrouped = dfc.groupby('visit')
    dfp = dfgrouped.agg(
            cscaled=('minmax', np.median),
            cmedian=(metric, np.median)
            ).reset_index()
    col_median = '{}_median'.format(metric)
    col_scaled = '{}_scaled'.format(metric)
    dfp.rename(columns={'cmedian': col_median, 'cscaled': col_scaled}, inplace=True)
    return dfp


def visits_plot_per_metric(df, x, y, hover_columns=None, filt=0):
    '''
    * x: name of the column for x-axis
    * y: name of the column for y-axis
    * hover_columns: list of column names for hover information
    '''
    from bokeh.models import HoverTool

    if hover_columns:
        _tt = [(n,'@{%s}' % n) for n in hover_columns]
        hover = HoverTool(tooltips=_tt)
    else:
        hover = 'hover'

    # 'x' must be renamed for Hv/Pn link axes with equal name/label;
    # in here, it will cause plots on different filter to have their
    # x-axis values merged (producing big blank areas in each plot)
    x_renamed = 'visits ({filt})'.format(filt=filt)
    df = df.sort_values(x)
    df[x_renamed] = df[x].astype(str)

    curve = hv.Curve(df, x_renamed, y)

    points = hv.Scatter(df, [x_renamed, y], hover_columns)
    points = points.opts(size=8, line_color='white',
                         tools=[hover])

    plot = curve * points
    plot = plot.redim(y=hv.Dimension(y, range=(-1, 1)))

    return plot
