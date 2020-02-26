import pytest
import os
from pathlib import Path
from lsst_dashboard.dataset import Dataset


@pytest.fixture()
def dataset():
    """
    Dataset object fixture
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'RC2_v18')
    d = Dataset(path=path)
    return d


def test_dataset_object():
    path = os.path.join(os.path.dirname(__file__), 'data', 'RC2_v18')
    d = Dataset(path=path)
    assert isinstance(d, Dataset)


def test_download_sample_data():
	"""test to ensure that CI has downloaded the sample data"""
	sample_data_folder = 'sample_data'
	# construct the path to the sample data
	path = Path().cwd().joinpath(sample_data_folder)
	# get the number of downloaded parquet files
	number_of_parq_files = len(list(path.glob('*.parq')))

	assert number_of_parq_files == 258
