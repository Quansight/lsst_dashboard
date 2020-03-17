#!/usr/bin/env python
import getpass
import socket
import os

from dask.distributed import Client, LocalCluster

hostname = socket.gethostname()
host = hostname.split('.')[0]
username = getpass.getuser()

# dask ports need to be between 29000 and 29999 (hard requirement due to firewall constraints)
# ssh forwarded ports should be between 20000 and 21000 (recommendation)

DASK_ALLOWED_PORTS = (29000, 30000)
DASHBOARD_ALLOWED_PORTS = (20000, 21000)
LOCAL_DASHBOARD = 5000
LOCAL_DASK_DASHBOARD = 8989

def find_available_ports(n, start, stop):
    count = 0
    for port in range(start, stop):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if count >= n:
                return
            try:
                res = sock.bind(('localhost', port))
                count += 1
                yield port
            except OSError:
                continue
        
def main():
    if 'lsst-dev' in host:
        from dask_jobqueue import SLURMCluster

        # dask ports need to be between 29000 and 29999 (hard requirement due to firewall constraints)
        scheduler_port, worker_port = find_available_ports(2, *DASK_ALLOWED_PORTS)

        # dashboard ports and jhub etc need to be between 20000 and 20999 (recommended no hard requirement)
        lsst_dashboard_port, dask_dashboard_port = find_available_ports(2, *DASHBOARD_ALLOWED_PORTS)

        print(f'starting dask cluster using slurm on {host}')
        cluster = SLURMCluster(
            queue='debug',
            cores=24,
            memory='128GB',
            scheduler_port=scheduler_port,
            extra=[f'--worker-port {worker_port}'],
            dashboard_address=f':{dask_dashboard_port}',
        )

        cluster.scale(2)
        client = Client(cluster)
        print('waiting for at least one node')
        client.wait_for_workers(1)

        # currently local and server ports need to match
        LOCAL_DASHBOARD = lsst_dashboard_port
        LOCAL_DASK_DASHBOARD = dask_dashboard_port

        print('starting dashboard')
        print('run the command below from your local machine to view dashboard:')
        print(f'\nssh -N -L {LOCAL_DASHBOARD}:{host}:{lsst_dashboard_port} -L {LOCAL_DASK_DASHBOARD}:{host}:{dask_dashboard_port} {username}@{hostname}\n')
    else:
        lsst_dashboard_port = 52001
        dask_dashboard_port = 52002

        LOCAL_DASHBOARD = lsst_dashboard_port
        LOCAL_DASK_DASHBOARD = dask_dashboard_port

        print(f'starting dask cluster on {host}')
        cluster = LocalCluster(dashboard_address=f':{LOCAL_DASK_DASHBOARD}')
        client = Client(cluster)

        
    print(f'### lsst dashboard available at http://localhost:{LOCAL_DASHBOARD} ###')
    print(f'### dask dashboard available at http://localhost:{LOCAL_DASK_DASHBOARD} ###')

    from lsst_dashboard.gui import dashboard

    # bokeh.server.views.ws - ERROR - Refusing websocket connection from Origin 'http://localhost:5000';                      
    # use --allow-websocket-origin=localhost:5000 or set BOKEH_ALL
    # os.environ["BOKEH_ALL"] = "" 
    # need to use same port on local and server for now.
    dashboard.render().show(port=lsst_dashboard_port)


if __name__ == "__main__":
    main()
