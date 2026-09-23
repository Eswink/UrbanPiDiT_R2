"""Bounded train-only controller fitting and exact parent-checkpoint restoration."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch.utils.data import default_collate
from model.r7_halting import AdaptiveProcessForecaster,positive_int
from .r7_halting import calibrate_controller_step
from .r7_experiment import make_model,load_checkpoint,dataset_identity,select_device,seed_everything,save_exclusive,model_code_digest,canonical_digest


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(module):
    digest=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):
        digest.update(key.encode()+b'\0')
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_calibration(parent_checkpoint,train_manifest,*,output_dir,updates=10,batch_size=1,
                    max_steps=4,hidden=64,gain_threshold=0.,probability_threshold=.5,
                    lr=1e-3,seed=42,device_name='cpu'):
    for value,name in [(updates,'updates'),(batch_size,'batch_size'),(max_steps,'max_steps'),(hidden,'hidden')]:
        positive_int(value,name)
    if max_steps<2 or not math.isfinite(lr) or lr<=0:
        raise ValueError('at least two steps and positive finite lr required')
    parent=load_checkpoint(parent_checkpoint)
    if parent['contract']['kind']!='process':
        raise ValueError('controller requires a process forecaster checkpoint')
    identity,ds=dataset_identity(train_manifest)
    if any(rec['split']!='train' for rec in ds.records):
        raise ValueError('calibration is training-only; validation/test records are forbidden')
    if identity!=parent['contract']['data_identity']:
        raise ValueError('parent training identity does not match calibration data')
    device=select_device(device_name)
    seed_everything(seed)
    forecaster=make_model('process',parent['contract']['model'])
    forecaster.load_state_dict(parent['model'],strict=True)
    forecaster=forecaster.to(device).eval().requires_grad_(False)
    before=state_digest(forecaster)
    adapter=AdaptiveProcessForecaster(forecaster,hidden,gain_threshold,probability_threshold).to(device)
    adapter.forecaster.eval()
    adapter.controller.train()
    optimizer=torch.optim.AdamW(adapter.controller.parameters(),lr=lr,weight_decay=1e-4)
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=False)
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    started=time.perf_counter()
    history=[]
    for step in range(updates):
        start=(step*batch_size)%len(ds)
        indices=[(start+i)%len(ds) for i in range(min(batch_size,len(ds)))]
        batch=default_collate([ds[i] for i in indices])
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        result=calibrate_controller_step(adapter,optimizer,batch,max_steps=max_steps)
        history.append({'update':step+1,'loss':float(result.total),'mean_target_gain':float(result.target_gains.mean())})
        del batch,result
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    if state_digest(forecaster)!=before:
        raise RuntimeError('frozen forecaster changed; controller checkpoint not published')
    contract={'parent_sha256':file_sha256(parent_checkpoint),'training_identity':identity,
        'model_code_sha256':model_code_digest(),'max_steps':max_steps,'hidden':hidden,
        'gain_threshold':float(gain_threshold),'probability_threshold':float(probability_threshold),
        'updates':updates,'batch_size':batch_size,'seed':seed,'lr':lr}
    payload={'format':'r7-controller-v1','contract':contract,'signature':canonical_digest(contract),
        'controller':adapter.controller.state_dict(),'optimizer':optimizer.state_dict()}
    path=folder/'controller.pt'
    save_exclusive(path,payload)
    report={'scientific_claim':False,'training_only':True,'forecaster_unchanged':True,
        'contract':contract,'elapsed_seconds':elapsed,'losses':history,
        'warning':'optimizer updates do not certify held-out calibration or speedup; thresholds are not tuned on test'}
    with (folder/'calibration.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    return path,report


def load_calibrated_adapter(forecaster,path,*,parent_checkpoint,training_identity):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!='r7-controller-v1':
        raise ValueError('unsupported controller checkpoint')
    c=saved['contract']
    if saved['signature']!=canonical_digest(c):
        raise ValueError('controller contract digest mismatch')
    if c['parent_sha256']!=file_sha256(parent_checkpoint):
        raise ValueError('controller parent checkpoint mismatch')
    if c['training_identity']!=training_identity or c['model_code_sha256']!=model_code_digest():
        raise ValueError('controller training/model identity mismatch')
    adapter=AdaptiveProcessForecaster(forecaster,c['hidden'],c['gain_threshold'],c['probability_threshold'])
    adapter.controller.load_state_dict(saved['controller'],strict=True)
    if adapter.controller.optimizer_updates.item()<1:
        raise ValueError('untrained controller checkpoint')
    return adapter.eval(),c
