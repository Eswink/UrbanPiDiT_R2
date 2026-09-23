"""Logical saved-tensor lifetime accounting, NOT an allocator/VRAM profiler."""
from __future__ import annotations

import torch


class _Packed:
    def __init__(self, tensor: torch.Tensor, meter: "SavedTensorMeter"):
        # Detach avoids retaining the original autograd tensor in the pack hook.
        self.tensor = tensor.detach()
        self.meter = meter
        self.nbytes = tensor.numel() * tensor.element_size()
        meter.live_bytes += self.nbytes
        meter.total_saved_bytes += self.nbytes
        meter.peak_live_bytes = max(meter.peak_live_bytes, meter.live_bytes)

    def __del__(self):
        self.meter.live_bytes -= self.nbytes


class SavedTensorMeter:
    """Count live logical bytes in autograd saved-tensor slots.

    Counts aliases and repeatedly saved weights separately. Excludes unsaved
    activations, allocator reserves, optimizer state and kernel workspaces.
    Useful for K-scaling regression; never label these numbers CUDA peak VRAM.
    """
    def __init__(self):
        self.live_bytes = 0
        self.peak_live_bytes = 0
        self.total_saved_bytes = 0
        self._hooks = None

    def __enter__(self):
        if self._hooks is not None:
            raise RuntimeError("meter cannot be reentered")
        self._hooks = torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: _Packed(tensor, self), lambda packed: packed.tensor)
        self._hooks.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self._hooks.__exit__(*args)
        finally:
            self._hooks = None
