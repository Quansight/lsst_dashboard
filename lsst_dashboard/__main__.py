#!/usr/bin/env python
import click
import getpass
import socket
import os

from dask.distributed import Client, LocalCluster

hostname = socket.gethostname()
host = hostname.split('.')[0]
username = getpass.getuser()

# dask ports need to be between 29000 and 29999 (hard requirement due to firewall constraints)
# ssh forwarded ports should be between 20000 and 21000 (recommendation)
# Using different ranges for dashboard and dask dashboard to avoid bad redirect behavior by 
# chrome when a dask dashboard port is reused as dashboard port 

DASK_ALLOWED_PORTS = (29000, 30000)
DASK_DASHBOARD_ALLOWED_PORTS = (20000, 20500)
DASHBOARD_ALLOWED_PORTS = (20500, 21000)

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


@click.command()
@click.option('--queue', default='debug', help='Slurm Queue to use (default=debug), ignored when on local machine')
@click.option('--nodes', default=2, help='Number of compute nodes to launch (default=2), ignored when on local machine')
@click.option('--localcluster', is_flag=True, help='Launches a localcluster instead of slurmcluster')
def main(queue, nodes, localcluster):
    if 'lsst-dev' in host and not localcluster:
        from dask_jobqueue import SLURMCluster

        scheduler_port, = find_available_ports(1, *DASK_ALLOWED_PORTS)
        lsst_dashboard_port, = find_available_ports(1, *DASHBOARD_ALLOWED_PORTS)   
        dask_dashboard_port, = find_available_ports(1, *DASK_DASHBOARD_ALLOWED_PORTS) 

        print(f'...starting dask cluster using slurm on {host} (queue={queue})')
        cluster = SLURMCluster(
            queue=queue,
            cores=24,
            processes=6,
            memory='128GB',
            scheduler_port=scheduler_port,
            extra=[f'--worker-port {":".join(str(p) for p in DASK_ALLOWED_PORTS)}'],
            dashboard_address=f':{dask_dashboard_port}',
        )

        print(f'...requesting {nodes} nodes')
        cluster.scale(nodes)
        client = Client(cluster)
        print('...waiting for at least one node')
        client.wait_for_workers(1)

        # currently local and server ports need to match
        LOCAL_DASHBOARD = lsst_dashboard_port
        LOCAL_DASK_DASHBOARD = dask_dashboard_port

        print('...starting dashboard')
        print('run the command below from your local machine to view dashboard:')
        print(f'\nssh -N -L {LOCAL_DASHBOARD}:{host}:{lsst_dashboard_port} -L {LOCAL_DASK_DASHBOARD}:{host}:{dask_dashboard_port} {username}@{hostname}\n')
    else:
        LOCAL_DASHBOARD = 52001
        LOCAL_DASK_DASHBOARD = 52002
        
        print(f'starting local dask cluster on {host}')
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
