from torch.utils.data import DataLoader
from data.synthetic import SyntheticR2Dataset
from training.lit_module import R2LightningModule


def _batch():
    return next(iter(DataLoader(
        SyntheticR2Dataset(
            length=4,coarse_hw=(8,8),urban_hw=(32,32)
        ),
        batch_size=2,
    )))


def _cfg():
    return dict(
        coarse_channels=12,coarse_history_steps=2,
        urban_channels=7,urban_history_steps=2,
        static_channels=6,out_channels=7,
        coarse_dim=64,process_dim=64,urban_dim=64,
        coarse_depth=1,urban_depth=1,
        coarse_heads=4,process_heads=4,urban_heads=4,
        window_size=4,free_processes=4,
    )


def test_training_step_backward():
    batch=_batch()
    lit=R2LightningModule(_cfg(),{'lr':1e-3},{'compute_weight':0.0})
    loss=lit.training_step(batch,0)
    loss.backward()
    assert loss.isfinite()


def test_verifier_has_training_signal():
    batch=_batch()
    lit=R2LightningModule(_cfg(),{'lr':1e-3},{'verifier_weight':0.2})
    out=lit.net(batch,force_zoom=True)
    loss=lit.loss_fn(batch,out)
    assert loss.verifier.isfinite() and loss.verifier.item() > 0


def test_process_deep_supervision_reaches_initializer_head():
    batch=_batch()
    lit=R2LightningModule(
        _cfg(),{'lr':1e-3},
        {'process_weight':1.0,'compute_weight':0.0}
    )
    out=lit.net(batch,force_zoom=True)
    loss=lit.loss_fn(batch,out)
    loss.total.backward()
    grad=lit.net.process_init.process_head[-1].weight.grad
    assert grad is not None
    assert grad.isfinite().all()
    assert grad.abs().sum() > 0
