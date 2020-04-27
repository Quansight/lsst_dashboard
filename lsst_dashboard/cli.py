#!/usr/bin/env python
import click
import getpass
import socket
import os

from distributed import Client

hostname = socket.gethostname()
host = hostname.split(".")[0]
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
                res = sock.bind(("localhost", port))
                count += 1
                yield port
            except OSError:
                continue



def launch_dask_cluster(queue, nodes, localcluster):
    # Launch Dask Cluster
    if 'lsst-dev' in host:
        # Set up allowed ports
        scheduler_port, = find_available_ports(1, *DASK_ALLOWED_PORTS)
        lsst_dashboard_port, = find_available_ports(1, *DASHBOARD_ALLOWED_PORTS)
        dask_dashboard_port, = find_available_ports(1, *DASK_DASHBOARD_ALLOWED_PORTS)
    else:
        localcluster = True
        lsst_dashboard_port = 52001
        dask_dashboard_port = 52002

    if not localcluster:
        from dask_jobqueue import SLURMCluster

        print(f"...starting dask cluster using slurm on {host} (queue={queue})")
        cluster = SLURMCluster(
            queue=queue,
            cores=24,
            processes=6,
            memory="128GB",
            scheduler_port=scheduler_port,
            extra=[f'--worker-port {":".join(str(p) for p in DASK_ALLOWED_PORTS)}'],
            dashboard_address=f":{dask_dashboard_port}",
        )

        print(f"...requesting {nodes} nodes")
        cluster.scale(nodes)
        print('run the command below from your local machine to forward ports for view dashboard and dask diagnostics:')
        print(f'\nssh -N -L {lsst_dashboard_port}:{host}:{lsst_dashboard_port} -L {dask_dashboard_port}:{host}:{dask_dashboard_port} {username}@{hostname}\n')
    else:
        from dask.distributed import LocalCluster
        print(f'starting local dask cluster on {host}')
        cluster = LocalCluster(dashboard_address=f':{dask_dashboard_port}')

    print(f'### dask dashboard available at http://localhost:{dask_dashboard_port} ###')
    return cluster, lsst_dashboard_port


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--queue', default='debug', help='Slurm Queue to use (default=debug), ignored on local machine')
@click.option('--nodes', default=2, help='Number of compute nodes to launch (default=2), ignored on local machine')
@click.option('--localcluster', is_flag=True, help='Launches a localcluster instead of slurmcluster, default on local machine')
def cli(ctx, queue, nodes, localcluster):
    """LSST Data Explorer Launch Script"""
    ctx.ensure_object(dict)
    ctx.obj.update({
        'queue': queue,
        'nodes': nodes,
        'localcluster': localcluster,
    })

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command('dask', short_help='Just Launch Dask Cluster')
@click.pass_context
def start_dask(ctx):
    #print(ctx.obj['queue'], ctx.obj['nodes'], ctx.obj['localcluster'])
    cluster, _ = launch_dask_cluster(ctx.obj['queue'], ctx.obj['nodes'], ctx.obj['localcluster'])
    print(f'Dask Cluster: {cluster}')
    print(f'Connect to cluster with "client = Client({cluster.scheduler_address})"')
    input('Press Any Key to Exit')


@cli.command('dashboard', short_help='Launch Visualization Dashboard (w/ Dask)')
@click.pass_context
def start_dashboard(ctx):
    cluster, lsst_dashboard_port = launch_dask_cluster(ctx.obj['queue'], ctx.obj['nodes'], ctx.obj['localcluster'])
    client = Client(cluster)
    print(f'Dask Cluster: {cluster}')
    print(f'Waiting for at least one worker')
    client.wait_for_workers(1)
    print(f'### starting lsst data explorer at http://localhost:{lsst_dashboard_port} ###')
    
    from lsst_dashboard.gui import dashboard
    dashboard.render().show(port=lsst_dashboard_port)


@cli.command('repartition', short_help='Prepare Butler Data for Vizualization (w/ Dask)')
@click.pass_context
@click.argument("butler_path")
@click.argument("destination_path", required=False)
def repartition(ctx, butler_path, destination_path):
    """Repartition a Butler Dataset for use with LSST Data Explorer Dashboard"""
    cluster, _ = launch_dask_cluster(ctx.obj['queue'], ctx.obj['nodes'], ctx.obj['localcluster'])
    client = Client(cluster)
    print(f'Dask Cluster: {cluster}')
    print(f'Waiting for at least one worker')
    client.wait_for_workers(1)

    print(f'### repartitioning data from {butler_path}')
    from lsst_dashboard.partition import CoaddForcedPartitioner, CoaddUnforcedPartitioner, VisitPartitioner

    if destination_path is None:
        destination_path = f"{butler_path}/ktk"

    print(f'...partitioned data will be written to {destination_path}')

    print('...partitioning coadd forced data')
    coadd_forced = CoaddForcedPartitioner(butler_path, destination_path)
    coadd_forced.partition()
    coadd_forced.write_stats()

    print('...partitioning coadd unforced data')
    coadd_unforced = CoaddUnforcedPartitioner(butler_path, destination_path)
    coadd_unforced.partition()
    coadd_unforced.write_stats()

    print('...partitioning visit data')
    visits = VisitPartitioner(butler_path, destination_path)
    visits.partition()
    visits.write_stats()

    print('...partitioning complete')


if __name__ == "__main__":
    cli(obj={})
