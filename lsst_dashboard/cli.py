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
    """
    Usage from script:
        from distributed import Client
        from lsst_dashboard.cli import launch_dask_cluster
        cluster, port = launch_dask_cluster('normal', 6, False)
        client = Client(cluster)
    """
    # Launch Dask Cluster
    if "lsst-dev" in host:
        # Set up allowed ports
        (scheduler_port,) = find_available_ports(1, *DASK_ALLOWED_PORTS)
        (lsst_dashboard_port,) = find_available_ports(1, *DASHBOARD_ALLOWED_PORTS)
        (dask_dashboard_port,) = find_available_ports(1, *DASK_DASHBOARD_ALLOWED_PORTS)
    else:
        localcluster = True
        lsst_dashboard_port = 52001
        dask_dashboard_port = 52002

    if not localcluster:
        from dask_jobqueue import SLURMCluster

        print(f"...starting dask cluster using slurm on {host} (queue={queue})")
        procs_per_node = 6
        cluster = SLURMCluster(
            queue=queue,
            cores=24,
            processes=procs_per_node,
            memory="128GB",
            scheduler_port=scheduler_port,
            extra=[f'--worker-port {":".join(str(p) for p in DASK_ALLOWED_PORTS)}'],
            dashboard_address=f":{dask_dashboard_port}",
        )

        print(f"...requesting {nodes} nodes")
        cluster.scale(nodes * procs_per_node)
        print(
            "run the command below from your local machine to forward ports for view dashboard and dask diagnostics:"
        )
        print(
            f"\nssh -N -L {lsst_dashboard_port}:{host}:{lsst_dashboard_port} -L {dask_dashboard_port}:{host}:{dask_dashboard_port} {username}@{hostname}\n"
        )
    else:
        from dask.distributed import LocalCluster

        print(f"starting local dask cluster on {host}")
        cluster = LocalCluster(dashboard_address=f":{dask_dashboard_port}")

    print(f"### dask dashboard available at http://localhost:{dask_dashboard_port} ###")
    return cluster, lsst_dashboard_port


@click.command()
@click.option(
    "--queue", default="debug", help="Slurm Queue to use (default=debug), ignored on local machine"
)
@click.option(
    "--nodes", default=2, help="Number of compute nodes to launch (default=2), ignored on local machine"
)
@click.option(
    "--localcluster",
    is_flag=True,
    help="Launches a localcluster instead of slurmcluster, default on local machine",
)
def start_dashboard(queue, nodes, localcluster):
    """
        Launches lsst_data_explorer with a Dask Cluster.
    """
    cluster, lsst_dashboard_port = launch_dask_cluster(queue, nodes, localcluster)
    client = Client(cluster)
    print(f"Dask Cluster: {cluster}")
    print(f"Waiting for at least one worker")
    client.wait_for_workers(1)
    print(f"### starting lsst data explorer at http://localhost:{lsst_dashboard_port} ###")

    from lsst_dashboard.gui import dashboard

    dashboard.render().show(port=lsst_dashboard_port)


@click.command()
@click.argument("butler_path")
@click.argument("destination_path", required=False)
@click.option("--sample_frac", default=None, type=float, help="sample dataset by fraction [0-1]")
@click.option("--num_buckets", default=8, help="number of buckets per partition")
@click.option("--chunk_by_filter", default=False)
@click.option("--chunk_dfs", default=False)
@click.option(
    "--queue", default="debug", help="Slurm Queue to use (default=debug), ignored on local machine"
)
@click.option(
    "--nodes", default=2, help="Number of compute nodes to launch (default=2), ignored on local machine"
)
@click.option(
    "--localcluster",
    is_flag=True,
    help="Launches a localcluster instead of slurmcluster, default on local machine",
)
def repartition(
    butler_path,
    destination_path,
    sample_frac,
    num_buckets,
    chunk_by_filter,
    chunk_dfs,
    queue,
    nodes,
    localcluster,
):
    """Repartitions a Butler Dataset for use with LSST Data Explorer using a Dask cluster"""
    cluster, _ = launch_dask_cluster(queue, nodes, localcluster)
    client = Client(cluster)
    print(f"Dask Cluster: {cluster}")
    print(f"Waiting for at least one worker")
    client.wait_for_workers(1)

    print(f"### repartitioning data from {butler_path}")
    from lsst_dashboard.partition import CoaddForcedPartitioner, CoaddUnforcedPartitioner, VisitPartitioner

    if destination_path is None:
        destination_path = f"{butler_path}/ktk"

    partition_kws = dict(chunk_by_filter=chunk_by_filter, chunk_dfs=chunk_dfs)

    print(f"...partitioned data will be written to {destination_path}")

    print("...partitioning coadd forced data")
    coadd_forced = CoaddForcedPartitioner(
        butler_path, destination_path, sample_frac=sample_frac, num_buckets=num_buckets
    )
    coadd_forced.partition(**partition_kws)
    coadd_forced.write_stats()

    print("...partitioning coadd unforced data")
    coadd_unforced = CoaddUnforcedPartitioner(
        butler_path, destination_path, sample_frac=sample_frac, num_buckets=num_buckets
    )
    coadd_unforced.partition(**partition_kws)
    coadd_unforced.write_stats()

    print("...partitioning visit data")
    visits = VisitPartitioner(
        butler_path, destination_path, sample_frac=sample_frac, num_buckets=num_buckets
    )
    visits.partition(**partition_kws)
    visits.write_stats()

    print("...partitioning complete")
