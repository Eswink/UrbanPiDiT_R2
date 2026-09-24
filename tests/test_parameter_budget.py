from model import UrbanPiDiTR2

def test_4090d_config_parameter_budget():
    m=UrbanPiDiTR2(
        coarse_channels=30,coarse_history_steps=4,urban_channels=7,urban_history_steps=4,
        static_channels=12,out_channels=7,coarse_dim=192,process_dim=256,urban_dim=256,
        coarse_depth=4,urban_depth=8,coarse_heads=6,process_heads=8,urban_heads=8,window_size=8)
    n=sum(p.numel() for p in m.parameters()); assert n < 45_000_000
