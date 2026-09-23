"""Bounded steady-input forward timing; no IO/target metrics in timed regions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import torch
from torch.utils.data import default_collate


def _positive(value,name,maximum):
    if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=maximum:
        raise ValueError(f'{name} must be an integer in [1,{maximum}]')


def _tensor_digest(items):
    result=hashlib.sha256()
    for name,tensor in sorted(items):
        value=tensor.detach().contiguous().cpu()
        result.update(str((name,tuple(value.shape),str(value.dtype))).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def profile_forward(model,batch,*,forward_kwargs=None,warmup=3,repetitions=10,precision='fp32'):
    """Time one resident batch per forward; validate outside the timed region.

    Run different policies in separate CLI processes to reduce allocator/order
    effects. The caller is responsible for using a representative checkpoint,
    identical input batches, hardware, precision and repeat counts.
    """
    _positive(warmup,'warmup',100)
    _positive(repetitions,'repetitions',1000)
    if precision not in ('fp32','bf16'):
        raise ValueError('only fp32 and bf16 autocast are supported')
    if any(module.training for module in model.modules()):
        raise ValueError('profile requires all modules in eval mode')
    history=batch['coarse_history']
    if history.ndim!=5 or min(history.shape)<1 or not history.is_floating_point() or not torch.isfinite(history).all():
        raise ValueError('finite nonempty floating [B,T,C,H,W] history required')
    device=history.device
    if device.type not in ('cpu','cuda'):
        raise ValueError('CPU/CUDA profiling only')
    if history.dtype!=torch.float32:
        raise ValueError('FP32 resident inputs required; BF16 uses autocast explicitly')
    for value in list(model.parameters())+list(model.buffers()):
        if value.device!=device or (value.is_floating_point() and value.dtype!=torch.float32):
            raise ValueError('model and inputs must share device and FP32 floating storage')
    if device.type=='cuda' and precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 unsupported; no fallback')
    inputs={'coarse_history':history.detach().clone()}
    if 'lead_time_hours' in batch:
        lead=batch['lead_time_hours']
        if not isinstance(lead,torch.Tensor) or lead.device!=device or not torch.isfinite(lead).all():
            raise ValueError('lead time must be a finite tensor on the input device')
        inputs['lead_time_hours']=lead.detach().clone()
    kwargs=dict(forward_kwargs or {})
    # Reject data-like extras in kwargs as well; only scalar inference controls.
    allowed={'reasoning_steps','max_steps','min_steps','force_full_depth'}
    if not set(kwargs)<=allowed or any(isinstance(v,torch.Tensor) for v in kwargs.values()):
        raise ValueError('only declared inference controls may be forwarded')
    model_before=_tensor_digest(model.state_dict().items())
    input_before=_tensor_digest(inputs.items())
    shape=(history.shape[0],history.shape[2],history.shape[3],history.shape[4])

    def synchronize():
        if device.type=='cuda':
            torch.cuda.synchronize(device)

    def call():
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
            return model(inputs,**kwargs)

    def check(result):
        prediction=result.forecast
        if prediction.shape!=shape or not torch.isfinite(prediction).all():
            raise ValueError('profile requires finite full-state forecasts matching input channels/grid')
        counts=getattr(result,'reasoning_steps_per_sample',None)
        if counts is None:
            k=int(getattr(result,'reasoning_steps',0))
            counts=torch.full((shape[0],),k,dtype=torch.long,device=device)
        if counts.shape!=(shape[0],) or counts.dtype not in (torch.int32,torch.int64) or (counts<0).any():
            raise ValueError('invalid actual reasoning-step counts')
        return prediction,counts.detach().cpu().tolist()

    for _ in range(warmup):
        output=call()
        synchronize()
        check(output)
        del output
    synchronize()
    baseline_allocated=baseline_reserved=None
    if device.type=='cuda':
        baseline_allocated=torch.cuda.memory_allocated(device)
        baseline_reserved=torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations=[]
    forward_peaks=[]
    counts=[]
    first=None
    maximum_difference=0.
    for _ in range(repetitions):
        synchronize()
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        output=call()
        synchronize()
        durations.append(time.perf_counter()-start)
        if device.type=='cuda':
            forward_peaks.append((torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)))
        prediction,actual=check(output)
        current=prediction.detach().float().cpu().clone()
        if first is None:
            first=current
        else:
            maximum_difference=max(maximum_difference,float((current-first).abs().max()))
        counts.append(actual)
        del current,prediction,output
    synchronize()
    peaks={'baseline_allocated_bytes':baseline_allocated,'baseline_reserved_bytes':baseline_reserved,
        'peak_allocated_bytes':max(p[0] for p in forward_peaks) if forward_peaks else None,
        'peak_reserved_bytes':max(p[1] for p in forward_peaks) if forward_peaks else None}
    if _tensor_digest(model.state_dict().items())!=model_before or _tensor_digest(inputs.items())!=input_before:
        raise RuntimeError('model or resident inputs mutated during profiling; no report accepted')
    properties=torch.cuda.get_device_properties(device) if device.type=='cuda' else None
    return {'format':'r7-inference-profile-v1','scientific_claim':False,
        'scope':'resident-batch model forward including Python/control and synchronization; excludes IO, transfer and validation/metrics',
        'device':str(device),'hardware':properties.name if properties else platform.processor() or platform.machine(),
        'total_device_memory_bytes':properties.total_memory if properties else None,
        'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'python':platform.python_version(),
        'cpu_threads':torch.get_num_threads(),'precision':precision,
        'cuda_matmul_tf32':bool(torch.backends.cuda.matmul.allow_tf32),'cudnn_tf32':bool(torch.backends.cudnn.allow_tf32),
        'input_shape':list(history.shape),'input_sha256':input_before,'model_state_sha256':model_before,
        'output_sha256':_tensor_digest([('forecast',first)]),'parameters':sum(p.numel() for p in model.parameters()),
        'warmup_excluded':warmup,'repetitions':repetitions,'forward_kwargs':kwargs,
        'seconds_per_batch':durations,'median_seconds_per_batch':statistics.median(durations),
        'mean_seconds_per_batch':statistics.mean(durations),'actual_reasoning_steps_per_sample':counts,
        'max_abs_repeat_difference':maximum_difference,'cuda_memory':peaks,
        'limitations':['one resident input batch is not a weather-distribution benchmark',
            'counts are not a speedup claim; compare actual timed runs under matched conditions',
            'CUDA peaks include model/input and warmed allocator; CPU fields are null, not GPU measurements',
            'run fixed/adaptive variants in fresh processes and repeat experiments to assess variability']}


def profile_local(manifest,*,checkpoint,output,controller_checkpoint=None,force_full_depth=False,
                  reasoning_steps=None,batch_size=1,warmup=3,repetitions=10,device_name='cpu',precision='fp32',policy_selection=None):
    from data.r7_zarr_dataset import ZarrAtmosWindowDataset
    from .r7_experiment import make_model,load_checkpoint,dataset_identity,select_device
    from .r7_calibration_runner import load_calibrated_adapter,file_sha256
    _positive(batch_size,'batch_size',64)
    _positive(warmup,'warmup',100)
    _positive(repetitions,'repetitions',1000)
    if precision not in ('fp32','bf16'):
        raise ValueError('unsupported precision')
    out=Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if policy_selection is not None and (not controller_checkpoint or force_full_depth or reasoning_steps is not None):
        raise ValueError('selected profiling policy requires controller and cannot be overridden')
    if force_full_depth and not controller_checkpoint:
        raise ValueError('force_full_depth requires a controller')
    saved=load_checkpoint(checkpoint)
    identity,_=dataset_identity(Path(manifest).parent/'train.jsonl')
    if identity!=saved['contract']['data_identity']:
        raise ValueError('checkpoint training data identity mismatch')
    ds=ZarrAtmosWindowDataset(manifest)
    if len(ds)<batch_size:
        raise ValueError('requested batch exceeds available manifest samples')
    model=make_model(saved['contract']['kind'],saved['contract']['model'])
    model.load_state_dict(saved['model'],strict=True)
    kwargs={}
    controller_hash=selection_hash=None
    effective_policy=None
    if controller_checkpoint:
        if saved['contract']['kind']!='process':
            raise ValueError('controller requires process checkpoint')
        model,policy=load_calibrated_adapter(model,controller_checkpoint,parent_checkpoint=checkpoint,training_identity=identity)
        if reasoning_steps is not None and reasoning_steps!=policy['max_steps']:
            raise ValueError('controller profiling must use its calibrated max_steps')
        kwargs={'max_steps':policy['max_steps'],'force_full_depth':bool(force_full_depth)}
        controller_hash=file_sha256(controller_checkpoint)
        effective_policy=dict(gain_threshold=float(model.gain_threshold),probability_threshold=float(model.probability_threshold),
            min_steps=1,max_steps=policy['max_steps'],force_full_depth=bool(force_full_depth))
        if policy_selection is not None:
            from .r7_policy_selection import load_selection
            effective_policy,selection_hash=load_selection(policy_selection,parent_sha256=file_sha256(checkpoint),
                controller_sha256=controller_hash,training_identity=identity)
            if effective_policy['max_steps']!=policy['max_steps']:
                raise ValueError('selection max_steps differs from calibrated depth')
            model.gain_threshold.fill_(effective_policy['gain_threshold'])
            model.probability_threshold.fill_(effective_policy['probability_threshold'])
            kwargs={k:effective_policy[k] for k in ('max_steps','min_steps','force_full_depth')}
    elif saved['contract']['kind']!='native':
        kwargs={'reasoning_steps':saved['contract']['steps'] if reasoning_steps is None else reasoning_steps}
    elif reasoning_steps is not None:
        raise ValueError('native forecaster has no reasoning_steps control')
    device=select_device(device_name,bf16=precision=='bf16')
    batch=default_collate([ds[i] for i in range(batch_size)])
    # Transfers are intentionally outside model-forward timing; no target is moved.
    inputs={k:batch[k].to(device) for k in ('coarse_history','lead_time_hours') if k in batch}
    report=profile_forward(model.to(device).eval(),inputs,forward_kwargs=kwargs,
        warmup=warmup,repetitions=repetitions,precision=precision)
    report.update(checkpoint_sha256=file_sha256(checkpoint),controller_sha256=controller_hash,
        training_identity=identity,evaluation_manifest_sha256=file_sha256(manifest),
        policy_selection_sha256=selection_hash,halting_policy=effective_policy,
        sample_ids=[r.get('sample_id') for r in ds.records[:batch_size]],
        source_declaration=ds._store(ds.records[0]).attrs['source'])
    with out.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
    return report
