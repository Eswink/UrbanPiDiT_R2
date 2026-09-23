import pytest
import torch
from training.r7_experiment import canonical_digest,save_exclusive,load_checkpoint,model_code_digest


def test_model_implementation_is_bound_to_checkpoint(tmp_path):
    contract={'kind':'native','model':{'in_channels':1}}
    payload={'format':'r7-local-v1','contract':contract,'signature':canonical_digest(contract)}
    path=tmp_path/'checkpoint.pt'
    save_exclusive(path,payload)
    saved=load_checkpoint(path)
    assert saved['model_code_sha256']==model_code_digest()
    saved['model_code_sha256']='wrong-revision'
    bad=tmp_path/'bad.pt'
    torch.save(saved,bad)
    with pytest.raises(ValueError,match='implementation'):
        load_checkpoint(bad)
