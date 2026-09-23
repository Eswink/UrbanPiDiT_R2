import argparse
from training.trainer import load_config, run

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); run(load_config(a.config))
if __name__=='__main__': main()
