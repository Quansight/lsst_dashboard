# from dask.distributed import Client
# client = Client()
from lsst_dashboard.gui import dashboard;
dashboard.render().servable()