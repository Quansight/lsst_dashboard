import pandas as pd
import numpy as np
import holoviews as hv
import panel as pn


def visits_plot(dsets_visits, filters_to_metrics, filt, errors=[], statistic="mean", selections={}):
    metrics = filters_to_metrics[filt]
    dset_filt = dsets_visits.loc[filt, :, :, statistic].reset_index(
        ["filter", "tract", "statistic"], drop=True
    )
    # there is one value per tract, take the median
    del dset_filt["visit"]
    dset_filt = dset_filt.groupby("visit").median().sort_index()
    return visits_plot_per_filter(dset_filt, metrics, filt, statistic, errors=errors, selections=selections)


def visits_plot_layout(plots):
    tabs = [(filt, pn.panel(plot)) for filt, plot in plots.items()]
    return pn.Tabs(*tabs, sizing_mode="stretch_both")


def visits_plot_per_filter(dsets_filter, metrics, filt, statistic, errors=[], selections={}):
    """
    * dsets_filter: a dictionary pointing to metrics dataframes
    """
    plot_all = None
    for metric in metrics:
        try:
            df = dsets_filter[metric].reset_index()
            col_stat = f"{metric}_{statistic}"
            df.rename(columns={metric: col_stat}, inplace=True)
            plot_metric = visits_plot_per_metric(df, 'visit',  col_stat,
                                                 [col_stat, 'visit'],
                                                 filt=filt)
            if (filt, metric) in selections:
                stream = selections[(filt, metric)]
                plot = plot_metric.get(1)
                stream.source = plot
                if stream.index and plot is not None:
                    plot_metric.get(1).opts(selected=stream.index)
            plot_all = plot_all * plot_metric if plot_all else plot_metric
        except:
            errors.append(metric)

    if plot_all:
        return plot_all.opts(
            aspect=5,
            bgcolor="black",
            show_legend=False,
            show_grid=True,
            gridstyle={"grid_line_color": "white", "grid_line_alpha": 0.2},
            responsive=True,
            xrotation=45,
            ylabel="normalized/metric",
        )
    else:
        return None


def visits_plot_per_metric(df, x, y, hover_columns=None, filt=0):
    """
    * x: name of the column for x-axis
    * y: name of the column for y-axis
    * hover_columns: list of column names for hover information
    """
    from bokeh.models import HoverTool
    from holoviews.core.util import dimension_sanitizer

    if hover_columns:
        _tt = [(n, "@{%s}" % dimension_sanitizer(n)) for n in hover_columns]
        hover = HoverTool(tooltips=_tt)
    else:
        hover = "hover"

    # 'x' must be renamed for Hv/Pn link axes with equal name/label;
    # in here, it will cause plots on different filter to have their
    # x-axis values merged (producing big blank areas in each plot)
    x_renamed = "visits ({filt})".format(filt=filt)
    df = df.sort_values(x)
    df[x_renamed] = df[x].astype(str)

    curve = hv.Curve(df, x_renamed, y)

    points = hv.Scatter(df, [x_renamed, y], hover_columns).opts(
        size=8, line_color="white", tools=[hover, "tap"], toolbar="above"
    )

    return (curve * points).redim(y=hv.Dimension(y, range=(-1, 1)))
