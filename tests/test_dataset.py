import pytest
import os
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
