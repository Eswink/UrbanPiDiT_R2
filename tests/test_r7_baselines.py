import pytest
import torch
from data.synthetic_atmos import SyntheticAtmosDataset
from model.r7_baselines import AFNOMixer
from model.r7_rollout import autoregressive_rollout
from training.r7_experiment import make_model,load_checkpoint
from training.r7_local_runner import run_local_updates
from training.r7_comparison_plan import write_comparison_plan


def config(architecture):
    cfg=dict(architecture=architecture,in_channels=2,history_steps=2,dim=16)
    if architecture=='afno_small':
        cfg.update(depth=1,blocks=4,shrinkage=0.)
    return cfg


@pytest.mark.parametrize('architecture',['unet','convlstm','afno_small'])
@pytest.mark.parametrize('hw',[(9,15),(4,6)])
def test_baseline_grid_gradients_and_no_target_dependence(architecture,hw):
    torch.manual_seed(7)
    m=make_model('native',config(architecture))
    history=torch.randn(2,2,2,*hw,requires_grad=True)
    a=m({'coarse_history':history})
    assert a.forecast.shape==(2,2,*hw)
    a.forecast.square().mean().backward()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    m.eval()
    with torch.no_grad():
        a=m({'coarse_history':history}).forecast
        b=m({'coarse_history':history,'atmos_target':torch.full_like(a,float('nan')),'urban_baseline':torch.full_like(a,999.)}).forecast
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    rollout=autoregressive_rollout(m,{'coarse_history':history},lead_hours=(6,12))
    assert rollout.forecasts.shape==(2,2,2,*hw)


def test_spectral_mixer_bf16_keeps_finite_fp32_fft_gradients():
    m=AFNOMixer(16,blocks=4,shrinkage=0.)
    x=torch.randn(2,16,5,7,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        y=m(x)
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert m.w1.grad is not None and m.w1.grad.abs().sum()>0


@pytest.mark.parametrize('architecture',['unet','convlstm','afno_small'])
def test_baseline_shared_runner_checkpoint_resume(tmp_path,architecture):
    ds=SyntheticAtmosDataset(length=5,hw=(8,12),channels=2)
    kwargs=dict(kind='native',model_config=config(architecture),data_identity='synthetic-baseline-test',
        output_dir=tmp_path/'run',batch_size=2,accumulation=2,steps=0,process_weight=0.,seed=9)
    first,_=run_local_updates(ds,total_updates=1,**kwargs)
    last,_=run_local_updates(ds,total_updates=2,resume=first,**kwargs)
    assert load_checkpoint(last)['updates']==2


def test_comparison_plan_is_explicit_and_non_executing(tmp_path):
    plan=write_comparison_plan(tmp_path/'plan',channels=2,seeds=(42,),dim=16)
    assert plan['state']=='PLANNED_NOT_RUN'
    assert len(plan['cases'])==9
    assert all(row['parameters']>0 for row in plan['cases'])
    assert not list((tmp_path/'plan').glob('*.pt'))
    with pytest.raises(FileExistsError):
        write_comparison_plan(tmp_path/'plan',channels=2,seeds=(42,),dim=16)
