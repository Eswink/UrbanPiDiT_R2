import base64
import struct
import numpy as np
import pytest
from test_r7_earthmover_plan import Array
from data.download.earthmover_pilot import DecodedBudget, bounded_selection, declared_missing_values


def test_pinned_provider_encoded_nan_remains_supported():
    a = Array(np.zeros((2,2),dtype='f4'), (2,2),attrs={'_FillValue':'AAAAAAAA+H8='})
    assert np.isnan(declared_missing_values(a)[0])
    np.testing.assert_array_equal(bounded_selection(a,(np.arange(2),np.arange(2)),DecodedBudget()),a.x)


def test_encoded_finite_sentinel_is_checked():
    encoded = base64.b64encode(struct.pack('<d',-9999.)).decode()
    a = Array(np.full((2,2),-9999.,dtype='f4'),(2,2),attrs={'_FillValue':encoded})
    with pytest.raises(ValueError,match='CF missing'):
        bounded_selection(a,(np.arange(2),np.arange(2)),DecodedBudget())


@pytest.mark.parametrize('encoded',['-9999','not-base64!','AAAA',''])
def test_invalid_encoding_refused_without_field_read(encoded):
    a = Array(np.ones((2,2),dtype='f4'),(2,2),attrs={'_FillValue':encoded})
    with pytest.raises(ValueError): bounded_selection(a,(np.arange(2),np.arange(2)),DecodedBudget())
    assert not a.calls
