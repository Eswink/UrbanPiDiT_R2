from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data import R7ForecastDataModule
from .r7_recursive_lit_module import R7RecursiveLightningModule


def load_recursive_config(path:str|Path)->Dict[str,Any]:
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_recursive(cfg:Dict[str,Any]):
    tc=cfg.get('train',{})
    pl.seed_everything(int(tc.get('seed',42)),workers=True)
    dm=R7ForecastDataModule(cfg.get('data',{}),tc)
    lit=R7RecursiveLightningModule(
        cfg.get('model',{}),
        cfg.get('optim',{}),
        cfg.get('loss',{}),
    )
    precision=tc.get(
        'precision',
        'bf16-mixed'
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else ('16-mixed' if torch.cuda.is_available() else '32-true'),
    )
    ck=ModelCheckpoint(
        dirpath=tc.get(
            'ckpt_dir','outputs/checkpoints/r7_recursive'
        ),
        monitor='val/loss',
        mode='min',
        save_top_k=int(tc.get('save_top_k',2)),
        save_last=True,
    )
    callbacks=[ck]
    if int(tc.get('patience',0))>0:
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                mode='min',
                patience=int(tc['patience']),
            )
        )
    trainer=pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=int(tc.get('max_epochs',10)),
        precision=precision,
        accumulate_grad_batches=int(
            tc.get('accumulate_grad_batches',1)
        ),
        gradient_clip_val=float(tc.get('gradient_clip_val',1.0)),
        limit_train_batches=tc.get('limit_train_batches',1.0),
        limit_val_batches=tc.get('limit_val_batches',1.0),
        limit_test_batches=tc.get('limit_test_batches',1.0),
        callbacks=callbacks,
        enable_progress_bar=bool(tc.get('progress_bar',True)),
        logger=False,
    )
    trainer.fit(lit,datamodule=dm)
    trainer.test(
        lit,
        datamodule=dm,
        ckpt_path='best' if ck.best_model_path else None,
    )
    return ck.best_model_path or ck.last_model_path
