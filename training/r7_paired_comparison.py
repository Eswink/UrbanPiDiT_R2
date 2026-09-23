"""Paired calendar-block uncertainty for pooled per-variable RMSE differences."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _rows(report):
    channels=report['channels']
    leads=report['lead_hours']
    if not channels or len(channels)!=len(set(channels)) or len(report['units'])!=len(channels):
        raise ValueError('unique variables and one unit per variable required')
    if not leads or sorted(set(leads))!=leads or any(isinstance(h,bool) or not isinstance(h,int) or h<1 for h in leads):
        raise ValueError('positive ordered unique lead hours required')
    rows={}
    for row in report['initializations']:
        init=pd.Timestamp(row['init_time'])
        if pd.isna(init) or init.tz is not None or init.isoformat() in rows:
            raise ValueError('invalid/duplicate initialization time')
        valid=[pd.Timestamp(v) for v in row['valid_times']]
        expected=[init+pd.Timedelta(hours=h) for h in leads]
        if valid!=expected:
            raise ValueError('valid times do not match requested leads')
        error=np.asarray(row['mse'],dtype=np.float64)
        if error.shape!=(len(leads),len(channels)) or not np.isfinite(error).all() or (error<0).any():
            raise ValueError('MSE must be finite nonnegative [lead,variable]')
        rows[init.isoformat()]=(init,error)
    if not rows or report.get('n_evaluated')!=len(rows):
        raise ValueError('nonempty initialization set/count required')
    return rows


def paired_rmse_bootstrap(a,b,*,block_days=7,replicates=1000,seed=42,confidence=.95):
    for value,name in [(block_days,'block_days'),(replicates,'replicates')]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(confidence) or not 0<confidence<1:
        raise ValueError('confidence must be in (0,1)')
    fields=['evaluation_manifest_sha256','channels','units','lead_hours','step_hours','split']
    for key in fields:
        if key not in a or a[key]!=b.get(key):
            raise ValueError(f'paired comparison metadata mismatch: {key}')
    ra,rb=_rows(a),_rows(b)
    if set(ra)!=set(rb):
        raise ValueError('initialization sets differ; no silent intersection')
    keys=sorted(ra)
    xa,xb=np.stack([ra[k][1] for k in keys]),np.stack([rb[k][1] for k in keys])
    block=np.array([ra[k][0].toordinal()//block_days for k in keys])
    unique=np.unique(block)
    if len(unique)<2:
        raise ValueError('at least two calendar blocks required; uncertainty is otherwise unsupported')
    sa=np.stack([xa[block==i].sum(0) for i in unique])
    sb=np.stack([xb[block==i].sum(0) for i in unique])
    counts=np.array([(block==i).sum() for i in unique])
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(replicates):
        sampled=rng.integers(0,len(unique),size=len(unique))
        n=counts[sampled].sum()
        draws.append(np.sqrt(sa[sampled].sum(0)/n)-np.sqrt(sb[sampled].sum(0)/n))
    draws=np.stack(draws)
    alpha=(1-confidence)/2
    low,high=np.quantile(draws,[alpha,1-alpha],axis=0)
    point=np.sqrt(xa.mean(0))-np.sqrt(xb.mean(0))
    result=[]
    for i,lead in enumerate(a['lead_hours']):
        for j,name in enumerate(a['channels']):
            result.append({'lead_hours':lead,'variable':name,'unit':a['units'][j],
                'rmse_A_minus_B':float(point[i,j]),'interval_lower':float(low[i,j]),'interval_upper':float(high[i,j])})
    return {'scientific_claim':False,'method':'paired nonoverlapping calendar-block percentile bootstrap',
        'difference_direction':'A minus B; negative means A has lower RMSE on supplied cases',
        'block_days':block_days,'n_blocks':len(unique),'replicates':replicates,'seed':int(seed),
        'confidence':float(confidence),'n_initializations':len(keys),'rows':result,
        'limitations':['uncertainty assumes chosen blocks adequately capture temporal dependence',
            'choose block length using validation/physical timescales, not favorable test intervals',
            'few blocks, multi-variable comparisons and model-selection bias need separate treatment',
            'these intervals are not a joint multiple-comparison test or SOTA certificate']}


def compare_files(path_a,path_b,*,out,**kwargs):
    output=Path(out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    text_a,text_b=Path(path_a).read_bytes(),Path(path_b).read_bytes()
    result=paired_rmse_bootstrap(json.loads(text_a),json.loads(text_b),**kwargs)
    result['input_sha256']=[hashlib.sha256(t).hexdigest() for t in (text_a,text_b)]
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    return result
