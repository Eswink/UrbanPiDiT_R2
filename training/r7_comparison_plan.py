"""Generate comparison configs and parameter counts, never launch experiments."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import yaml
from .r7_experiment import make_model


def write_comparison_plan(out_dir,*,channels=17,seeds=(42,43,44),dim=128):
    if channels<1 or dim<16 or dim%8 or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('invalid channel/dimension/seed plan')
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=False)
    common={'in_channels':channels,'out_channels':channels,'history_steps':2}
    window=dict(common,dim=dim,depth=4,heads=4,window_size=8)
    cases=[
        ('unet','native',dict(common,architecture='unet',dim=max(16,dim//2)),0.),
        ('convlstm','native',dict(common,architecture='convlstm',dim=dim),0.),
        ('afno_small','native',dict(common,architecture='afno_small',dim=dim,depth=4,blocks=4),0.),
        ('window','native',window,0.),
        ('window_deep','native',dict(window,depth=8),0.),
        ('generic','generic',dict(window,latent_tokens=16),0.),
        ('process_no_feedback','process',dict(window,anchored_processes=8,free_processes=8,use_forecast_feedback=False),.1),
        ('process_no_aux','process',dict(window,anchored_processes=8,free_processes=8),0.),
        ('process','process',dict(window,anchored_processes=8,free_processes=8),.1),
    ]
    rows=[]
    for name,kind,model_config,process_weight in cases:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model=make_model(kind,model_config)
            params=sum(p.numel() for p in model.parameters())
            del model
        for seed in seeds:
            train={'seed':int(seed),'batch_size':1,'accumulation':4,'steps':0 if kind=='native' else 4,
                'process_weight':process_weight,'lr':2e-4}
            config={'kind':kind,'model':model_config,'train':train}
            path=out/f'{name}_seed{seed}.yaml'
            path.write_text(yaml.safe_dump(config,sort_keys=True),encoding='utf-8')
            rows.append({'case':name,'seed':int(seed),'config':path.name,'parameters':params,
                'evaluation_K':[0] if kind=='native' else [1,2,4,8]})
    plan={'state':'PLANNED_NOT_RUN','cases':rows,
        'required':'Use one immutable dataset/split/variable order/normalization, same search budget and real measured compute.',
        'not_claimed':['equal FLOPs','equal parameter budgets','SOTA','real weather results'],
        'extra_controls':['persistence via evaluate_r7_local --persistence','trained adaptive controller vs forced full depth']}
    (out/'comparison_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan
