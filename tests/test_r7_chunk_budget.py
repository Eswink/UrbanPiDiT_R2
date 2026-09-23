import pytest

from data.download.chunk_budget import (
    ChunkBudgetExceeded,
    estimate_bundle_chunk_bytes,
    estimate_variable_chunk_bytes,
)


TIMES=[0,1,2,100,101,102,200,201,202]
ROI={"latitude":slice(200,221),"longitude":slice(460,481)}


def wb2_pressure():
    return {
        "dims":["time","level","latitude","longitude"],
        "shape":[92044,13,721,1440],
        "chunks":[1,13,721,1440],
        "dtype":"float32",
    }


def ncar_pressure():
    return {
        "dimensions":["time","level","latitude","longitude"],
        "shape":[24,37,721,1440],
        "chunks":[1,1,181,360],
        "dtype":"float32",
    }


def test_tiny_roi_still_counts_whole_weatherbench_pressure_chunks():
    row=estimate_variable_chunk_bytes(
        wb2_pressure(),
        {"time":TIMES,"level":[7,9],**ROI},
    )
    assert row["touched_chunks"]==9
    assert row["uncompressed_bytes_per_full_chunk"]==13*721*1440*4
    assert row["estimated_uncompressed_bytes"]>450*1024*1024


def test_bundle_cap_rejects_whole_globe_pressure_slab_source():
    surface={
        "dims":["time","latitude","longitude"],
        "shape":[92044,721,1440],
        "chunks":[1,721,1440],
        "dtype":"float32",
    }
    variables={f"s{i}":surface for i in range(4)}
    variables.update({f"p{i}":wb2_pressure() for i in range(4)})
    selections={}
    for i in range(4):
        selections[f"s{i}"]={"time":TIMES,**ROI}
        selections[f"p{i}"]={"time":TIMES,"level":[7,9],**ROI}
    with pytest.raises(ChunkBudgetExceeded):
        estimate_bundle_chunk_bytes(
            {"variables":variables},selections,max_bytes=512*1024*1024
        )


def test_spatial_level_chunking_makes_same_logical_selection_small():
    row=estimate_variable_chunk_bytes(
        ncar_pressure(),
        {"time":list(range(9)),"level":[20,30],
         "latitude":slice(200,221),"longitude":slice(460,481)},
    )
    assert row["touched_chunks"]<=72
    assert row["estimated_uncompressed_bytes"]<32*1024*1024


def test_contiguous_unknown_and_implicit_full_dimensions_fail_closed():
    meta=wb2_pressure()
    bad=dict(meta,chunks=None)
    with pytest.raises(ValueError,match="unsafe"):
        estimate_variable_chunk_bytes(
            bad,{"time":[0],"level":[0],"latitude":slice(0,1),"longitude":slice(0,1)}
        )
    with pytest.raises(ValueError,match="explicit selection"):
        estimate_variable_chunk_bytes(
            meta,{"time":[0],"level":[0],"latitude":slice(0,1)}
        )
