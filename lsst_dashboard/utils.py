"collection of utilities"
from pathlib import Path

import asyncio


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
    tracts = list(set([int(t.name.split('-')[-1]) for t in folder.rglob('*tract*')]))
    filters = [f.name for f in folder.joinpath('plots').iterdir() if f.is_dir()]

    return tracts, filters
