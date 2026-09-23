from data.synthetic import SyntheticR2Dataset
from data.schema import validate_sample

def test_synthetic_contract():
    s=SyntheticR2Dataset(length=2,coarse_hw=(8,8),urban_hw=(32,32))[0]; validate_sample(s,batched=False); assert s['urban_target'].shape[-2:]==(32,32)
