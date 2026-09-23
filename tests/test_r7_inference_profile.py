from types import SimpleNamespace
import pytest
import torch
from training.r7_inference_profile import profile_forward


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(1))
        self.calls=0
        self.keys=[]

    def forward(self,batch,**kwargs):
        self.calls+=1
        self.keys.append(set(batch))
        result=batch['coarse_history'][:,-1]*self.weight
        return SimpleNamespace(forecast=result,reasoning_steps=kwargs.get('reasoning_steps',0))


def batch():
    return dict(coarse_history=torch.ones(2,2,3,4,5),lead_time_hours=torch.tensor([6.,6.]),
        atmos_target=torch.full((2,3,4,5),float('nan')),process_targets=torch.tensor([float('nan')]))


def test_forward_timing_excludes_warmup_and_targets():
    model=Tiny().eval()
    inputs=batch()
    before=inputs['coarse_history'].clone()
    result=profile_forward(model,inputs,warmup=2,repetitions=3,forward_kwargs={'reasoning_steps':4})
    assert model.calls==5
    assert all(k=={'coarse_history','lead_time_hours'} for k in model.keys)
    assert len(result['seconds_per_batch'])==3 and all(x>0 for x in result['seconds_per_batch'])
    assert result['actual_reasoning_steps_per_sample']==[[4,4]]*3
    assert result['max_abs_repeat_difference']==0.
    assert all(v is None for v in result['cuda_memory'].values())
    assert result['scientific_claim'] is False
    torch.testing.assert_close(inputs['coarse_history'],before)


@pytest.mark.parametrize('keyword,value',[('warmup',0),('warmup',True),('repetitions',0),('repetitions',1001),('precision','fp16')])
def test_invalid_profile_controls(keyword,value):
    with pytest.raises(ValueError):
        profile_forward(Tiny().eval(),batch(),**{keyword:value})


def test_mode_dtype_and_kwarg_guards():
    with pytest.raises(ValueError,match='eval'):
        profile_forward(Tiny(),batch(),warmup=1,repetitions=1)
    b=batch(); b['coarse_history']=b['coarse_history'].half()
    with pytest.raises(ValueError,match='FP32'):
        profile_forward(Tiny().eval(),b,warmup=1,repetitions=1)
    with pytest.raises(ValueError,match='inference controls'):
        profile_forward(Tiny().eval(),batch(),forward_kwargs={'atmos_target':torch.zeros(1)},warmup=1,repetitions=1)


class Mutating(Tiny):
    def forward(self,batch,**kwargs):
        batch['coarse_history'].add_(1.)
        return super().forward(batch,**kwargs)


def test_input_mutation_rejected_without_changing_callers_batch():
    b=batch(); original=b['coarse_history'].clone()
    with pytest.raises(RuntimeError,match='mutated'):
        profile_forward(Mutating().eval(),b,warmup=1,repetitions=1)
    torch.testing.assert_close(b['coarse_history'],original)


class BadOutput(Tiny):
    def forward(self,batch,**kwargs):
        out=super().forward(batch,**kwargs)
        out.forecast.fill_(float('nan'))
        return out


def test_nonfinite_prediction_fails():
    with pytest.raises(ValueError,match='finite full-state'):
        profile_forward(BadOutput().eval(),batch(),warmup=1,repetitions=1)


def test_cpu_bf16_records_precision_without_cuda_claim():
    result=profile_forward(Tiny().eval(),batch(),warmup=1,repetitions=1,precision='bf16')
    assert result['precision']=='bf16' and result['device']=='cpu'
    assert result['total_device_memory_bytes'] is None
