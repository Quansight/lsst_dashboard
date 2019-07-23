"collection of utilities"
from pathlib import Path


def scan_folder(path):
    """Given a folder return available tracts and filters
    """
    folder = Path(path)
    tracts = list(set([int(t.name.split('-')[-1]) for t in folder.rglob('*tract*')]))
    filters = [f.name for f in folder.joinpath('plots').iterdir() if f.is_dir()]

    return tracts, filters
