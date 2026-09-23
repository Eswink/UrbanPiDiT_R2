from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.r7_comparison_plan import write_comparison_plan

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--channels',type=int,default=17)
    ap.add_argument('--dim',type=int,default=128)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    a=ap.parse_args()
    plan=write_comparison_plan(a.out,channels=a.channels,seeds=a.seeds,dim=a.dim)
    print(f"Wrote {len(plan['cases'])} planned configs; no training launched.")
