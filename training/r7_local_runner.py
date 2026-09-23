"""Bounded single-device runner. No downloads, implicit datasets or GPU fallback."""
from __future__ import annotations
import json
import math
from pathlib import Path
import platform
import time
import torch
from torch.utils.data import default_collate
from model.r7_halting import forecast_inputs
from .r7_halting import per_sample_latitude_mse
from .r7_streaming import backward_streamed_truncated
from .r7_experiment import canonical_digest, make_model, select_device, seed_everything, rng_state, restore_rng, save_exclusive, load_checkpoint


def _integer(value,name,minimum=1):
    if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def update_group(model,optimizer,cpu_batches,*,kind,device,steps,bf16,process_weight,clip):
    """Move one microbatch at a time; one optimizer step per sample-weighted group."""
    count = sum(len(b['coarse_history']) for b in cpu_batches)
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    try:
        for cpu_batch in cpu_batches:
            batch = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in cpu_batch.items()}
            scale = len(batch['coarse_history'])/count
            if kind == 'native':
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
                    prediction = model(forecast_inputs(batch)).forecast
                loss = per_sample_latitude_mse(prediction,batch['atmos_target'],batch.get('latitude')).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite native training loss')
                (loss*scale).backward()
                value = float(loss.detach())
                del prediction,loss
            else:
                result = backward_streamed_truncated(model,batch,reasoning_steps=steps,
                    process_weight=process_weight,loss_scale=scale,amp_dtype=torch.bfloat16 if bf16 else None)
                value = float(result.total)
                del result
            total += value*scale
            del batch
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total


def run_local_updates(dataset,*,kind,model_config,data_identity,output_dir,total_updates=10,
                      batch_size=1,accumulation=1,steps=4,seed=42,lr=2e-4,
                      process_weight=.1,clip=1.,device_name='cpu',bf16=False,resume=None):
    """Run until an explicit optimizer-update endpoint; checkpoint at that endpoint.

    Resume is deterministic on the same tested hardware/software for deterministic
    datasets. No cross-platform bitwise promise. Full-BPTT Lightning stays separate.
    """
    for value,name in [(total_updates,'total_updates'),(batch_size,'batch_size'),(accumulation,'accumulation')]:
        _integer(value,name)
    _integer(steps,'steps',0)
    _integer(seed,'seed',0)
    if len(dataset)<1 or not math.isfinite(lr) or lr<=0 or not math.isfinite(clip) or clip<=0:
        raise ValueError('nonempty dataset and positive finite optimizer settings required')
    if not math.isfinite(process_weight) or process_weight<0:
        raise ValueError('invalid process loss weight')
    device=select_device(device_name,bf16)
    contract={'kind':kind,'model':dict(model_config),'data_identity':str(data_identity),
        'batch_size':batch_size,'accumulation':accumulation,'steps':steps,'seed':seed,
        'lr':lr,'process_weight':process_weight,'clip':clip,'bf16':bool(bf16),
        'device_type':device.type,'torch_version':str(torch.__version__),
        'dataset_length':len(dataset),'optimization':'streamed-truncated' if kind!='native' else 'native'}
    signature=canonical_digest(contract)
    saved=load_checkpoint(resume,expected=signature) if resume else None
    folder=Path(output_dir)
    checkpoint_path=folder/f'update_{total_updates:07d}.pt'
    metrics_path=folder/f'updates_to_{total_updates:07d}.json'
    if checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError('requested update endpoint already published')
    if saved and saved['updates']>=total_updates:
        raise ValueError('resume endpoint must be greater than saved updates')
    if not saved:
        folder.mkdir(parents=True,exist_ok=False)
    else:
        folder.mkdir(parents=True,exist_ok=True)
    seed_everything(seed)
    model=make_model(kind,model_config).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    updates,epoch,cursor=0,0,0
    if saved:
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        updates,epoch,cursor=saved['updates'],saved['epoch'],saved['cursor']
        if not (0<=cursor<=len(dataset)):
            raise ValueError('checkpoint cursor outside dataset')
        restore_rng(saved['rng'])
    losses=[]
    if device.type=='cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    while updates<total_updates:
        if cursor==len(dataset):
            epoch+=1
            cursor=0
        order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        batches=[]
        for _ in range(accumulation):
            if cursor==len(dataset):
                break
            indices=order[cursor:min(cursor+batch_size,len(dataset))]
            batches.append(default_collate([dataset[i] for i in indices]))
            cursor+=len(indices)
        first=batches[0]
        expected_channels=model_config.get('out_channels') or model_config['in_channels']
        if first['atmos_target'].shape[1]!=expected_channels:
            raise ValueError('dataset/model channel mismatch')
        if kind=='process' and process_weight>0 and 'process_targets' not in first:
            raise ValueError('process training requires diagnostic targets; explicitly set process_weight=0 for an ablation')
        loss=update_group(model,optimizer,batches,kind=kind,device=device,steps=steps,
            bf16=bf16,process_weight=process_weight,clip=clip)
        updates+=1
        losses.append({'update':updates,'epoch':epoch,'loss':loss,'samples':sum(len(b['coarse_history']) for b in batches)})
        del batches,first
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-started
    report={'scientific_claim':False,'data_identity':str(data_identity),'contract':contract,
        'updates_this_run':len(losses),'elapsed_seconds':elapsed,'losses':losses,
        'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor() or 'CPU',
        'cuda_version':torch.version.cuda,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
        'note':'wall time includes batch reads; CUDA peaks cover this bounded run, not a forecast-skill benchmark'}
    payload={'format':'r7-local-v1','signature':signature,'contract':contract,
        'model':model.state_dict(),'optimizer':optimizer.state_dict(),'updates':updates,
        'epoch':epoch,'cursor':cursor,'rng':rng_state()}
    save_exclusive(checkpoint_path,payload)
    with metrics_path.open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    return checkpoint_path,report
