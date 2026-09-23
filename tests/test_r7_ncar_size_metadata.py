import numpy as np
import pytest
from data.download.ncar_public_probe import object_size_bytes


@pytest.mark.parametrize('info', [{'size': 31}, {'Size': 31}, {'ContentLength': 31}, {'size': np.int64(31)}, {'size': 31, 'Size': 31}])
def test_ncar_size_normalization(info):
    assert object_size_bytes(info) == 31


@pytest.mark.parametrize('info', [{}, {'size': None}, {'size': -1}, {'size': 0}, {'size': True}, {'size': 1.5}, {'size': '31'}, {'size': 31, 'Size': 32}])
def test_ncar_size_invalid_metadata(info):
    with pytest.raises(ValueError):
        object_size_bytes(info)
