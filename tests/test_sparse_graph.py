import inspect, torch
from model.sparse_process_graph import SparseGridProcessGraph

def test_sparse_graph_shape_and_no_cdist():
    g=SparseGridProcessGraph(32); x=torch.randn(2,64,32); y=g(x,(8,8)); assert y.shape==x.shape
    src=inspect.getsource(SparseGridProcessGraph); assert 'cdist' not in src; assert '[N, N]' not in src
