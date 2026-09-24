from __future__ import annotations
import sys, time, argparse, yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from model import UrbanPiDiTR2
from training.losses import R2Loss

PRICES={'5090':2.78*0.95,'v100':1.88*0.95}

def make_batch(cfg,batch_size,device):
    m=cfg['model']; B=batch_size; Hc=Wc=32; Hu=Wu=128
    def r(*shape): return torch.randn(*shape,device=device)
    return {
        'coarse_history':r(B,m['coarse_history_steps'],m['coarse_channels'],Hc,Wc),
        'urban_history':r(B,m['urban_history_steps'],m['urban_channels'],Hu,Wu),
        'urban_static':r(B,m['static_channels'],Hu,Wu),
        'urban_baseline':r(B,m['out_channels'],Hu,Wu),
        'urban_target':r(B,m['out_channels'],Hu,Wu),
        'process_targets':r(B,m.get('anchored_processes',12)),
        'zoom_target':torch.ones(B,device=device),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/r2_v6_4090d.yaml')
    ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--gpu',choices=['5090','v100','local'],default='local')
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('该 benchmark 需要 CUDA GPU。')
    cfg=yaml.safe_load(open(args.config,encoding='utf-8')); device='cuda'; B=int(cfg['train'].get('batch_size',1))
    model=UrbanPiDiTR2(**cfg['model']).to(device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    loss_fn=R2Loss(**cfg.get('loss',{}))
    batch=make_batch(cfg,B,device)
    use_bf16=cfg['train'].get('precision','').startswith('bf16'); dtype=torch.bfloat16 if use_bf16 else torch.float16
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.warmup+args.steps):
        if i==args.warmup: torch.cuda.synchronize(); t0=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):
            out=model(batch,force_zoom=True); loss=loss_fn(batch,out).total
        loss.backward(); opt.step()
    torch.cuda.synchronize(); dt=time.perf_counter()-t0; sps=args.steps/dt; peak=torch.cuda.max_memory_allocated()/1024**3
    print(f'steps/s={sps:.4f}  sec/step={1/sps:.4f}  peak_allocated={peak:.2f}GB')
    if args.gpu in PRICES:
        cost_1k=(1000/sps)/3600*PRICES[args.gpu]; print(f'{args.gpu} 会员价估算: ¥{cost_1k:.2f} / 1000 steps')
if __name__=='__main__': main()
