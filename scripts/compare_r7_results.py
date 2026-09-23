from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.r7_paired_comparison import compare_files

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--a',required=True)
    ap.add_argument('--b',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--block-days',type=int,default=7)
    ap.add_argument('--replicates',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    r=compare_files(a.a,a.b,out=a.out,block_days=a.block_days,replicates=a.replicates,seed=a.seed)
    print(f"Compared {r['n_initializations']} matched initializations in {r['n_blocks']} blocks.")
