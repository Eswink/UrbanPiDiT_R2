from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data import R7ForecastDataModule
from .r7_lit_module import R7ForecastLightningModule


def load_r7_config(path:str|Path)->Dict[str,Any]:
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_r7(cfg:Dict[str,Any]):
    train_cfg=cfg.get('train',{})
    seed=int(train_cfg.get('seed',42))
    pl.seed_everything(seed,workers=True)

    dm=R7ForecastDataModule(cfg.get('data',{}),train_cfg)
    lit=R7ForecastLightningModule(
        cfg.get('model',{}),
        cfg.get('optim',{}),
    )
    precision=train_cfg.get(
        'precision',
        'bf16-mixed'
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else ('16-mixed' if torch.cuda.is_available() else '32-true'),
    )
    ck=ModelCheckpoint(
        dirpath=train_cfg.get(
            'ckpt_dir','outputs/checkpoints/r7_native'
        ),
        monitor='val/loss',
        mode='min',
        save_top_k=int(train_cfg.get('save_top_k',2)),
        save_last=True,
    )
    callbacks=[ck]
    if int(train_cfg.get('patience',0))>0:
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                mode='min',
                patience=int(train_cfg['patience']),
            )
        )

    trainer=pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=int(train_cfg.get('max_epochs',10)),
        precision=precision,
        accumulate_grad_batches=int(
            train_cfg.get('accumulate_grad_batches',1)
        ),
        gradient_clip_val=float(train_cfg.get('gradient_clip_val',1.0)),
        log_every_n_steps=int(train_cfg.get('log_every_n_steps',10)),
        limit_train_batches=train_cfg.get('limit_train_batches',1.0),
        limit_val_batches=train_cfg.get('limit_val_batches',1.0),
        limit_test_batches=train_cfg.get('limit_test_batches',1.0),
        callbacks=callbacks,
        enable_progress_bar=bool(train_cfg.get('progress_bar',True)),
        logger=False,
    )
    trainer.fit(lit,datamodule=dm)
    trainer.test(
        lit,
        datamodule=dm,
        ckpt_path='best' if ck.best_model_path else None,
    )
    return ck.best_model_path or ck.last_model_path
