import pytest
import torch
from data import validate_forecast_sample


def test_r7_contract_rejects_spatial_mismatch():
    sample={
        'coarse_history':torch.randn(2,4,8,8),
        'atmos_target':torch.randn(4,7,8),
    }
    with pytest.raises(ValueError):
        validate_forecast_sample(sample,batched=False)


def test_r7_contract_accepts_metadata():
    sample={
        'coarse_history':torch.randn(2,4,8,12),
        'atmos_target':torch.randn(4,8,12),
        'lead_time_hours':torch.tensor(6.0),
        'latitude':torch.linspace(60,25,8),
        'longitude':torch.linspace(100,130,12),
    }
    validate_forecast_sample(sample,batched=False)
