"""Fixed tiny ARCO request with hard wire/object/decode bounds; no fallback data."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request
import numpy as np

BASE='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/'


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('redirects are not permitted for fixed-source smoke')


class BoundedObjects:
    def __init__(self,max_bytes=64*2**20,max_objects=64,deadline_seconds=240):
        if max_bytes<1 or max_bytes>64*2**20 or max_objects<1 or max_objects>64:
            raise ValueError('budgets must stay within hard smoke limits')
        self.max_bytes,self.max_objects=max_bytes,max_objects
        self.bytes=0
        self.receipts=[]
        self.deadline=time.monotonic()+deadline_seconds
        self.opener=urllib.request.build_opener(_NoRedirect)

    def read(self,key):
        if key.startswith('/') or '..' in key or '?' in key or '://' in key:
            raise ValueError('invalid fixed-source object key')
        if len(self.receipts)>=self.max_objects or time.monotonic()>self.deadline:
            raise RuntimeError('object/time budget exhausted')
        url=BASE+key
        request=urllib.request.Request(url,headers={'User-Agent':'UrbanPiDiT-R7-bounded-smoke/1','Accept-Encoding':'identity'})
        with self.opener.open(request,timeout=20) as response:
            if response.status!=200 or response.geturl()!=url:
                raise ValueError('unexpected source response')
            length=response.headers.get('Content-Length')
            if length is None:
                raise ValueError('Content-Length required to enforce strict byte budget')
            length=int(length)
            if length<0 or length>16*2**20 or length>self.max_bytes-self.bytes:
                raise RuntimeError('object/wire byte budget exceeded before reading')
            remaining=length
            chunks=[]
            while remaining:
                if time.monotonic()>self.deadline:
                    raise RuntimeError('download deadline reached')
                data=response.read(min(65536,remaining))
                if not data:
                    raise ValueError('truncated source object')
                self.bytes+=len(data)
                remaining-=len(data)
                chunks.append(data)
            payload=b''.join(chunks)
            self.receipts.append({'url':url,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
                'etag':response.headers.get('ETag'),'generation':response.headers.get('x-goog-generation')})
            return payload


def decode_chunk(payload,metadata):
    from numcodecs import get_codec
    dtype=np.dtype(metadata['dtype'])
    shape=tuple(metadata['chunks'])
    if dtype.kind not in 'fiu' or not shape or any(not isinstance(v,int) or v<1 for v in shape):
        raise ValueError('unsupported chunk dtype/shape')
    if math.prod(shape)*dtype.itemsize>16*2**20:
        raise ValueError('decoded chunk exceeds 16 MiB limit')
    if metadata.get('filters'):
        raise ValueError('filtered arrays need a reviewed decoder, not a guessed transform')
    if metadata.get('order','C') not in ('C','F'):
        raise ValueError('unknown array order')
    compressor=metadata.get('compressor')
    out=np.empty(shape,dtype=dtype,order=metadata.get('order','C'))
    if compressor is None:
        if len(payload)!=out.nbytes:
            raise ValueError('raw chunk byte count mismatch')
        out[...] = np.frombuffer(payload,dtype=dtype).reshape(shape,order=metadata.get('order','C'))
    else:
        if compressor.get('id') not in {'blosc','zstd','zlib','gzip'}:
            raise ValueError('unsupported reviewed compressor')
        get_codec(compressor).decode(payload,out=out)
    return out


def _array_1d(fetch,meta,name):
    spec=meta[name+'/.zarray']
    attrs=meta[name+'/.zattrs']
    if len(spec['shape'])!=1 or attrs.get('_ARRAY_DIMENSIONS')!=[name]:
        raise ValueError('coordinate schema changed')
    n=spec['shape'][0]
    if n<1 or n*np.dtype(spec['dtype']).itemsize>16*2**20:
        raise ValueError('coordinate exceeds bounded size')
    chunk=spec['chunks'][0]
    values=[]
    for i in range((n+chunk-1)//chunk):
        values.append(decode_chunk(fetch.read(f'{name}/{i}'),spec))
    return np.concatenate(values)[:n],attrs


def download_fixed_t2m(out_dir,fetch=None):
    import pandas as pd
    import xarray as xr
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=False)
    fetch=BoundedObjects() if fetch is None else fetch
    try:
        consolidated=json.loads(fetch.read('.zmetadata'))
        if consolidated.get('zarr_consolidated_format')!=1:
            raise ValueError('unsupported consolidated metadata version')
        meta=consolidated['metadata']
        spec=meta['2m_temperature/.zarray']
        attrs=meta['2m_temperature/.zattrs']
        if attrs.get('_ARRAY_DIMENSIONS')!=['time','latitude','longitude'] or attrs.get('units')!='K':
            raise ValueError('t2m dimensions/units differ from reviewed schema')
        if spec['chunks']!=[1,721,1440] or spec['shape'][1:]!=[721,1440]:
            raise ValueError('upstream surface chunking changed; no unbounded fallback')
        lat,_=_array_1d(fetch,meta,'latitude')
        lon,_=_array_1d(fetch,meta,'longitude')
        rawtime,timeattrs=_array_1d(fetch,meta,'time')
        timevar=xr.DataArray(rawtime,dims=['time'],attrs={k:v for k,v in timeattrs.items() if k!='_ARRAY_DIMENSIONS'})
        times=pd.DatetimeIndex(xr.decode_cf(xr.Dataset({'time':timevar}))['time'].values)
        if times.has_duplicates or times.hasnans or not times.is_monotonic_increasing:
            raise ValueError('invalid source time coordinate')
        chosen=pd.DatetimeIndex([f'{year}-01-01T{hour:02d}:00:00' for year in [2018,2019,2020] for hour in [0,6,12]])
        indices=times.get_indexer(chosen)
        if (indices<0).any():
            raise ValueError('requested historical frames unavailable')
        yi=np.flatnonzero((lat>=39.5)&(lat<=40.5))
        xi=np.flatnonzero((lon>=115.5)&(lon<=117.0))
        if len(yi)!=5 or len(xi)!=7:
            raise ValueError('unexpected 0.25-degree ROI grid')
        separator=spec.get('dimension_separator','.')
        if separator not in ('.','/'):
            raise ValueError('unsupported chunk key format')
        frames=[]
        for i in indices:
            key='2m_temperature/'+separator.join([str(i),'0','0'])
            full=decode_chunk(fetch.read(key),spec)
            frames.append(full[0][np.ix_(yi,xi)])
        data=np.stack(frames).astype(np.float32)
        if not np.isfinite(data).all():
            raise ValueError('source ROI contains nonfinite weather')
        ds=xr.Dataset({'2m_temperature':(('time','latitude','longitude'),data)},
            coords={'time':chosen,'latitude':lat[yi],'longitude':lon[xi]})
        ds['2m_temperature'].attrs={'units':'K','source':'ECMWF ERA5 via Google ARCO-ERA5'}
        ds.attrs['attribution']='Contains modified Copernicus Climate Change Service information 2026; distribution via ARCO-ERA5.'
        ds.to_netcdf(out/'source.nc',engine='h5netcdf')
        result={'status':'downloaded-real-source','source':BASE,'wire_bytes':fetch.bytes,'objects':fetch.receipts,
            'timestamps':[t.isoformat() for t in chosen],'shape':list(data.shape),'units':'K',
            'field_min':float(data.min()),'field_max':float(data.max()),'scientific_training_ready':False,
            'purpose':'9-frame one-variable geographic/data-path smoke only, not a forecast benchmark',
            'attribution':ds.attrs['attribution'],'reference':'https://github.com/google-research/arco-era5'}
    except Exception as exc:
        result={'status':'failed-no-fallback','error':f'{type(exc).__name__}: {exc}',
            'wire_bytes':fetch.bytes,'objects':fetch.receipts,'synthetic_fallback':False}
        (out/'receipts.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        raise
    (out/'receipts.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return out/'source.nc'
