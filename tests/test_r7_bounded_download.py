import io
import json
import numpy as np
import pytest
from data.download.arco_tiny_bounded import BASE,BoundedObjects,decode_chunk,download_fixed_t2m


class Response(io.BytesIO):
    status=200
    def __init__(self,payload,url,declared=None):
        super().__init__(payload)
        self.url=url
        self.headers={'Content-Length':str(len(payload) if declared is None else declared),'ETag':'test-etag'}
        self.read_calls=0
    def read(self,*args):
        self.read_calls+=1
        return super().read(*args)
    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self,response): self.response=response
    def open(self,*args,**kwargs): return self.response


def test_wire_cap_checked_before_any_body_read():
    fetch=BoundedObjects(max_bytes=4)
    response=Response(b'12345',BASE+'x')
    fetch.opener=FakeOpener(response)
    with pytest.raises(RuntimeError,match='budget'): fetch.read('x')
    assert response.read_calls==0 and fetch.bytes==0


def test_object_receipt_and_object_budget():
    fetch=BoundedObjects(max_bytes=4,max_objects=1)
    fetch.opener=FakeOpener(Response(b'1234',BASE+'x'))
    assert fetch.read('x')==b'1234'
    assert fetch.bytes==4 and fetch.receipts[0]['bytes']==4
    with pytest.raises(RuntimeError,match='budget'): fetch.read('x')
    with pytest.raises(ValueError): fetch.read('../private')


def test_decode_schema_and_fixed_size():
    meta={'dtype':'<f4','chunks':[1,3,4],'compressor':None,'order':'C'}
    values=np.arange(12,dtype=np.float32).reshape(1,3,4)
    np.testing.assert_array_equal(decode_chunk(values.tobytes(),meta),values)
    with pytest.raises(ValueError,match='16 MiB'):
        decode_chunk(b'',dict(meta,chunks=[1,721,1440,37]))
    with pytest.raises(ValueError,match='filters|filtered'):
        decode_chunk(values.tobytes(),dict(meta,filters=[{'id':'unknown'}]))


def test_failed_remote_read_never_creates_weather(tmp_path):
    class Offline:
        bytes=0
        receipts=[]
        def read(self,key): raise OSError('offline fixture')
    with pytest.raises(OSError): download_fixed_t2m(tmp_path/'attempt',fetch=Offline())
    report=json.loads((tmp_path/'attempt'/'receipts.json').read_text())
    assert report['status']=='failed-no-fallback' and report['synthetic_fallback'] is False
    assert not (tmp_path/'attempt'/'source.nc').exists()
