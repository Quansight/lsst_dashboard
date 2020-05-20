import param
import panel as pn
import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts
import geopandas as gpd
from shapely.geometry import box
from importlib import resources

pn.extension()


def debug(msg):
    # print(msg)
    pass


class OverviewApp(param.Parameterized):

    metrics_path = param.String("PDR2_metrics.parq")
    skymap_path = param.String("deepCoadd_skyMap.csv")
    hidden_tract = param.String("")

    metric = param.ObjectSelector()
    filter_ = param.ObjectSelector()

    selected_tract_str = param.String(default="", label="Selected tracts")

    plot_width = param.Integer(default=800, bounds=(1, None))
    plot_height = param.Integer(default=400, bounds=(1, None))

    def __init__(self, tracts_update_callback, active_tracts=None, **params):
        debug("OverviewApp.__init__()")
        # Declare Tap stream (source polygon element to be defined later)
        self.stream = hv.streams.Selection1D()
        super().__init__(**params)
        # set up default empty objects
        self.df = pd.DataFrame()
        self.rangexy = hv.streams.RangeXY()
        self.tracts_update_callback = tracts_update_callback
        self._active_tracts = active_tracts

        # load the skmap and metrics data
        self.load_data()

    @param.output()
    def output(self):
        debug("OverviewApp.output()")
        """output list of selected tracts"""
        # return an empty list if no tracts selected
        if self.stream.index == [""]:
            return list()
        else:
            # return the tract list from the widget (which matches the selection)
            return self.tracts_in_widget()

    def load_data(self):
        debug("OverviewApp.load_data()")

        """load in the source files, reorganize data, and set up widget options"""
        data_package = "lsst_dashboard.data"
        # load the metrics data
        with resources.path(data_package, self.metrics_path) as path:
            self.metrics_df = pd.read_parquet(path)

        # load skymap (csv with x0, x1, y0, y1 columns)
        with resources.path(data_package, self.skymap_path) as path:
            self.skymap = gpd.read_file(path).astype(float)

        self.skymap.geometry = self.skymap.apply(lambda x: box(x["x0"], x["y0"], x["x1"], x["y1"]), axis=1)

        # combine the metrics with the skymap geometry
        self.df = self.metrics_df.reset_index().rename(columns={"level_0": "filter", "level_1": "tract"})
        self.df = self.df.join(self.skymap, on="tract")

        # get the available metrics
        metrics = [c for c in self.metrics_df.columns if "_unit" not in c]
        metrics.sort()
        # set up the metric widget
        self.param.metric.objects = metrics
        self.metric = metrics[0]

        # set up the filter widget
        filters = list(set(self.df["filter"]))
        filters.sort()
        self.param.filter_.objects = filters
        self.filter_ = self.param.filter_.objects[0]

    def update_tract_selection(self, event):
        debug("OverviewApp.update_tract_selection()")
        """When tracts are added to the widget, select them on the screen"""
        # create a dataframe of the polygon data (for easy indexing)
        df = pd.DataFrame(self.polys.data)

        # get the selected tracts as list of ints
        tract_list = self.tracts_in_widget()

        # convert tracts to poly element indices
        poly_indices = list(
            df[df.apply(lambda x: True if x["tract"] in tract_list else False, axis=1)].index
        )
        # select the polygons in the plot
        self.stream.event(index=poly_indices)

    @property
    def active_tracts(self):
        debug("OverviewApp.active_tracts.getter()")

        if not self._active_tracts:
            return np.unique(self.df.tract).tolist()

        return self._active_tracts

    @active_tracts.setter
    def active_tracts(self, tracts):
        debug("OverviewApp.active_tracts.setter()")
        self._active_tracts = tracts
        self.hidden_tract = "".join([str(s) for s in tracts])

    def tracts_in_widget(self):
        debug("OverviewApp.tracts_in_widget()")
        """convert the tracts in the widget to a list of ints"""
        # get the selected tract strings
        selected_list = self.selected_tract_str.replace(" ", "").split(",")
        # convert tracts to list of ints
        tract_list = list([x if x == "" else int(x) for x in selected_list])

        if self.active_tracts is not None and len(self.active_tracts) > 0:
            return list(set(tract_list).intersection(set(self.active_tracts)))
        else:
            return tract_list

    @param.depends("stream.index", watch=True)
    def update_tract_widget(self):
        debug("OverviewApp.update_tract_widget()")
        """When tracts are selected on the map, display the tract numbers in the widget"""
        tract_dict = self.polys.iloc[self.stream.index].data
        self.selected_tract_str = ",".join([str(t["tract"]) for t in tract_dict])
        self.tracts_update_callback(self.tracts_in_widget())

    @param.depends("metric", "filter_", "hidden_tract")
    def plot(self):

        debug("OverviewApp.plot()")
        """plot pane"""
        # get the tract list
        tract_list = self.tracts_in_widget()

        # extract the filter from the original data
        self.df_filter = self.df[self.df["filter"] == self.filter_]

        # extract only the provided metric
        self.df_extracted = self.df_filter[["tract", "geometry", self.metric, "x0", "x1", "y0", "y1"]]

        # filter down tracts
        self.df_extracted = self.df_extracted.loc[self.df_extracted.tract.isin(self.active_tracts)]

        debug("OverviewApp.plot() --> Number of Tracts to Plot {}".format(len(self.df_extracted)))

        # rename the metric column (necessary abstraction for plotting)
        self.df_extracted = self.df_extracted.rename(columns={self.metric: "metric"})

        if self.df.empty:
            return pn.Spacer()
        else:

            def create_dict(row):
                """create a dictionary representation of a df row"""
                d = {
                    ("x", "y"): np.array(row.geometry.exterior.coords),
                    "tract": row["tract"],
                    "metric": row["metric"],
                }
                return d

            # create a dictionary reprresentation of the dataframe
            data = list(self.df_extracted.apply(create_dict, axis=1))

            # declare polygons
            self.polys = hv.Polygons(data, vdims=["metric", "tract"]).opts(
                tools=["hover", "tap", "box_select"],
                line_width=0,
                active_tools=["wheel_zoom", "tap"],
                colorbar=True,
                title=self.metric,
                color="metric",
            )

            # Declare Tap stream with polys as source and initial values
            self.stream.source = self.polys

            # Define a RangeXY stream linked to the image (preserving ranges from the previous image)
            self.rangexy = hv.streams.RangeXY(
                source=self.polys, x_range=self.rangexy.x_range, y_range=self.rangexy.y_range,
            )
            # set padding (degrees)
            padding = 0

            # get the limits of the selected filter/metric data
            xmin = self.df_extracted.x0.min() - padding
            xmax = self.df_extracted.x1.max() + padding
            ymin = self.df_extracted.y0.min() - padding
            ymax = self.df_extracted.y1.max() + padding

            # create hook for reseting to full extent
            def reset_range_hook(plot, element):
                plot.handles["x_range"].reset_start = xmin
                plot.handles["x_range"].reset_end = xmax
                plot.handles["y_range"].reset_start = ymin
                plot.handles["y_range"].reset_end = ymax

            # Define function to compute selection based on tap location
            def tap_histogram(index, x_range, y_range):
                (x0, x1), (y0, y1) = x_range, y_range
                # if nothing is selected
                if index == list():
                    # select all
                    index = list(self.df_extracted.index)
                return self.polys.iloc[index].opts(opts.Polygons(xlim=(x0, x1), ylim=(y0, y1)))

            tap_dmap = hv.DynamicMap(tap_histogram, streams=[self.stream, self.rangexy])

            # set up the plot to match the range stream
            if not self.rangexy.x_range and not self.rangexy.y_range:
                (x0, x1, y0, y1) = (None, None, None, None)
            else:
                (x0, x1), (y0, y1) = self.rangexy.x_range, self.rangexy.y_range

            return self.polys.opts(
                opts.Polygons(
                    xlim=(x0, x1), ylim=(y0, y1), hooks=[reset_range_hook], responsive=True, bgcolor="black"
                )
            )  # TODO: responsive=True isn't changing the width

    def left_pane(self):
        debug("OverviewApp.left_pane()")
        load_tracts = pn.widgets.Button(name="\u25b6", width=40)
        load_tracts.on_click(self.update_tract_selection)

        pane = pn.Row(self.param.metric, self.param.filter_, self.param.selected_tract_str,)
        return pane

    def panel(self):
        debug("OverviewApp.panel()")
        return pn.Column(
            self.left_pane,
            pn.panel(self.plot, sizing_mode="stretch_both", width_policy="max", height_policy="max"),
            sizing_mode="stretch_both",
        )


def create_overview(tracts_update_callback, active_tracts=None):
    overview_app = OverviewApp(tracts_update_callback, active_tracts=active_tracts)
    return overview_app
