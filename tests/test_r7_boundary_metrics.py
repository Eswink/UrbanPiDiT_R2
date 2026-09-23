import csv
import pytest
import torch
from training.r7_boundary_metrics import BoundaryRMSEAccumulator,boundary_masks


def sample():
    torch.manual_seed(12)
    return torch.randn(3,2,2,7,9),torch.zeros(3,2,2,7,9),torch.linspace(60.,30.,7)


def test_boundary_squared_error_decomposition_and_reverse_latitude():
    p,t,lat=sample()
    a=BoundaryRMSEAccumulator((6,12),('t','u'),margins=(1,2),training_std=(2.,3.),units=('K','m/s'))
    a.update(p,t,lat)
    values=a.compute().square()
    for index in (1,3):
        inside=a.regions[index]['full_area_fraction']
        edge=a.regions[index+1]['full_area_fraction']
        assert inside+edge==pytest.approx(1.)
        torch.testing.assert_close(values[0],inside*values[index]+edge*values[index+1])
    reverse=BoundaryRMSEAccumulator((6,12),('t','u'),margins=(1,2),training_std=(2.,3.),units=('K','m/s'))
    reverse.update(p.flip(-2),t.flip(-2),lat.flip(0))
    torch.testing.assert_close(reverse.compute(),a.compute())


def test_edge_only_error_and_uneven_batches(tmp_path):
    p,t,lat=sample()
    p.zero_()
    masks=boundary_masks(7,9,(1,))
    p[...,masks[2][2]]=5.
    a=BoundaryRMSEAccumulator((6,12),('t','u'),margins=(1,))
    a.update(p[:1],t[:1],lat)
    a.update(p[1:],t[1:],lat)
    values=a.compute()
    torch.testing.assert_close(values[1],torch.zeros_like(values[1]))
    torch.testing.assert_close(values[2],torch.full_like(values[2],5.))
    torch.testing.assert_close(values[0],torch.full_like(values[0],5.*a.regions[2]['full_area_fraction']**.5))
    assert a.initializations==3
    path=tmp_path/'boundary.csv'
    a.write_csv(path)
    with path.open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==12 and rows[0]['region']=='full'
    with pytest.raises(FileExistsError):
        a.write_csv(path)


@pytest.mark.parametrize('margins',[(0,),(-1,),(4,),(True,),(1,1),(),(1.5,)])
def test_invalid_empty_or_duplicate_margins(margins):
    with pytest.raises(ValueError):
        boundary_masks(7,9,margins)


def test_failures_do_not_mutate_statistics():
    p,t,lat=sample()
    a=BoundaryRMSEAccumulator((6,12),('t','u'))
    with pytest.raises(RuntimeError):
        a.compute()
    a.update(p,t,lat)
    before=a.compute().clone()
    bad=p.clone(); bad[0,0,0,0,0]=float('nan')
    for args in [(bad,t,lat),(p,t,lat+1),(p,t,torch.zeros(7)),(p,t,lat.repeat(3,1)+torch.arange(3)[:,None])]:
        with pytest.raises(ValueError):
            a.update(*args)
        torch.testing.assert_close(a.compute(),before)
        assert a.initializations==3


def test_unit_and_horizon_validation():
    for kw in [dict(units=('K',)),dict(training_std=(0.,)),dict(training_std=(1.,)),dict(training_std=(1.,),units=('',))]:
        with pytest.raises(ValueError):
            BoundaryRMSEAccumulator((6,),('t',),**kw)
    with pytest.raises(ValueError):
        BoundaryRMSEAccumulator((12,6),('t',))
