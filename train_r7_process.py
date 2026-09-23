from __future__ import annotations
import argparse

from training.r7_process_forecast_trainer import (
    load_process_forecast_config,
    run_process_forecast,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/r7_process_forecast_smoke.yaml",
    )
    args=ap.parse_args()
    path=run_process_forecast(
        load_process_forecast_config(args.config)
    )
    print(f"R7 process-forecast checkpoint: {path}")


if __name__=="__main__":
    main()
