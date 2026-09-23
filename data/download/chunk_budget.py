"""Conservative source-chunk byte budgeting before public ERA5 reads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numpy as np


class ChunkBudgetExceeded(RuntimeError):
    pass


def _meta_axes(meta: Mapping) -> tuple[list[str], list[int], list[int], np.dtype]:
    dims=meta.get("dims",meta.get("dimensions"))
    shape=meta.get("shape")
    chunks=meta.get("chunks")
    if dims is None or shape is None:
        raise ValueError("variable metadata must include dims/dimensions and shape")
    dims=list(dims)
    shape=[int(v) for v in shape]
    if chunks is None:
        raise ValueError("contiguous/unreported chunking is unsafe for bounded remote reads")
    chunks=[int(v) for v in chunks]
    if not (len(dims)==len(shape)==len(chunks)) or len(dims)<1:
        raise ValueError("dims, shape and chunks must have equal nonzero rank")
    if len(set(dims))!=len(dims) or any(v<1 for v in shape+chunks):
        raise ValueError("invalid dimension/chunk metadata")
    try:
        dtype=np.dtype(meta["dtype"])
    except Exception as exc:
        raise ValueError("valid dtype required") from exc
    return dims,shape,chunks,dtype


def _chunk_ids(selection, size: int, chunk: int) -> set[int]:
    if isinstance(selection,bool):
        raise ValueError("boolean is not a valid index selection")
    if isinstance(selection,int):
        index=selection+size if selection<0 else selection
        if not 0<=index<size:
            raise IndexError(index)
        return {index//chunk}
    if isinstance(selection,slice):
        start,stop,step=selection.indices(size)
        if step!=1:
            raise ValueError("only contiguous step=1 slices are supported for budgeting")
        if start>=stop:
            raise ValueError("empty selections are unsupported")
        return set(range(start//chunk,(stop-1)//chunk+1))
    if isinstance(selection,Sequence) and not isinstance(selection,(str,bytes,bytearray)):
        ids=set()
        seen=False
        for value in selection:
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)):
                raise ValueError("index sequences must contain integers")
            seen=True
            index=int(value)
            index=index+size if index<0 else index
            if not 0<=index<size:
                raise IndexError(index)
            ids.add(index//chunk)
        if not seen:
            raise ValueError("empty selections are unsupported")
        return ids
    raise TypeError("selection must be int, integer sequence or contiguous slice")


def estimate_variable_chunk_bytes(meta: Mapping, selections: Mapping[str, object]) -> dict:
    """Estimate whole uncompressed source chunks touched by an exact selection."""
    dims,shape,chunks,dtype=_meta_axes(meta)
    extra=set(selections).difference(dims)
    if extra:
        raise ValueError(f"unknown selected dimensions: {sorted(extra)}")
    missing=set(dims).difference(selections)
    if missing:
        raise ValueError(f"explicit selection required for dimensions: {sorted(missing)}")
    per_dim={}
    touched=1
    for name,size,chunk in zip(dims,shape,chunks):
        ids=_chunk_ids(selections[name],size,chunk)
        count=len(ids)
        per_dim[name]={"chunks_touched":count,"chunk_ids":sorted(ids)}
        touched*=count
    chunk_elements=math.prod(chunks)
    chunk_bytes=int(chunk_elements*dtype.itemsize)
    return {
        "touched_chunks":int(touched),
        "uncompressed_bytes_per_full_chunk":chunk_bytes,
        "estimated_uncompressed_bytes":int(touched*chunk_bytes),
        "per_dimension":per_dim,
        "dtype":str(dtype),
    }


def estimate_bundle_chunk_bytes(
    report: Mapping,
    selections_by_variable: Mapping[str,Mapping[str,object]],
    *,
    max_bytes: int | None=None,
) -> dict:
    variables=report.get("variables")
    if not isinstance(variables,Mapping):
        raise ValueError("report must contain variable metadata")
    if not selections_by_variable:
        raise ValueError("at least one variable selection is required")
    rows={}
    total=0
    for name,selections in selections_by_variable.items():
        if name not in variables:
            raise KeyError(name)
        row=estimate_variable_chunk_bytes(variables[name],selections)
        rows[name]=row
        total+=row["estimated_uncompressed_bytes"]
    result={"variables":rows,"estimated_uncompressed_bytes":int(total)}
    if max_bytes is not None:
        if isinstance(max_bytes,bool) or not isinstance(max_bytes,int) or max_bytes<1:
            raise ValueError("max_bytes must be a positive integer")
        result["max_bytes"]=max_bytes
        result["within_budget"]=total<=max_bytes
        if total>max_bytes:
            raise ChunkBudgetExceeded(
                f"estimated source chunks require {total} bytes > cap {max_bytes}"
            )
    return result
