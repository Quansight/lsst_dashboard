"collection of utilities"
from pathlib import Path

import asyncio

import panel as pn
import holoviews as hv
import datetime


def set_timeout(seconds, callback):
    async def schedule():

        await asyncio.sleep(seconds)

        if asyncio.iscoroutinefunction(callback):
            await callback()
        else:
            callback()

    asyncio.ensure_future(schedule())


def scan_folder(path):
    """Given a folder return available tracts and filters
    """
    folder = Path(path)
    tracts = list(set([int(t.name.split("-")[-1]) for t in folder.rglob("*tract*")]))
    filters = [f.name for f in folder.joinpath("plots").iterdir() if f.is_dir()]

    return tracts, filters


def clear_dynamicmaps(obj):
    """
    Clears all DynamicMaps on an object to force recalculation.
    """
    print(f"{datetime.datetime.now()}: clear_dynamicmaps...")

    for p in obj.select(pn.pane.HoloViews):
        for dmap in p.object.traverse(lambda x: x, [hv.DynamicMap]):
            dmap.data.clear()

    print(f"{datetime.datetime.now()}: end clear_dynamicmaps.")
