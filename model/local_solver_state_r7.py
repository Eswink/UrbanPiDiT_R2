"""RW-B: a per-position gated solver state, an anchored proposal and source roles.

RW-A gave every output position its own read of the process tokens but kept the
solver itself additive and stateless: the read is *added* to the context, the
correction is *added* to the current draft, and no position remembers anything
between internal steps. Three consequences are what RW-B exists to change, and
``docs/R7_MAIN_MODEL_V2_DESIGN.md`` section 3 is the authority for the equations:

    Z_(k+1)    = LocalUpdate(Z_k, C, E(Y_k), R_k, step_embedding)     # O(N), small kernel
    Y_proposal = X_t + Decoder(Z_(k+1))
    Y_(k+1)    = Y_k + g_k * (Y_proposal - Y_k),   g_k = sigmoid(.) in (0, 1)

``X_t`` is the known state at the current physical time - the persistence base the
backbone already decoded from - so the proposal is an *absolute* estimate relative
to the same anchor at every step instead of a tendency accumulating on top of the
previous draft. The gate decides per position how far to move towards it.

Three boundaries are deliberate, and each is asserted by a test rather than
promised in prose:

- **The gate may not cut the gradient the proposal needs.** Its bias initializes
  the sigmoid near 0.25, never at saturation, and the tests assert that the
  proposal side of the blend carries a nonzero gradient through ``g``.
  A sigmoid gate is a stabilizer for a candidate mechanism; it is *not* a claim
  of monotone improvement or physical correctness, and it must not be written as
  one.
- **The role markers are not a second mechanism.** The recurring cell reads
  ``cat([C, E(Y_k)])`` with nothing marking which half is which - the gap the
  design contract records in section 3.2. Adding a learned vector per half costs
  2 x dim parameters and is checked by one criterion: reversing the draft tokens
  must change the output when the markers are on and leave it (bit for bit)
  unchanged when they are off.
- **Nothing here reads a future field.** The decoder sees ``Z`` and the anchor;
  targets, future diagnostics and baselines never reach these modules.

The E0 diagnostics (``docs/R7_E0_DIAGNOSTICS.md``) are why the gate is on the
correction *amplitude*: adding a process read enlarged the solver's corrections
consistently across seeds (C/B ratio 1.31) without making them any less aligned
with the error. The gate is the mechanism that lets the model decline a large
proposal per position; it is not evidence that the proposal is right.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers.patch_grid import crop_native_grid

# sigmoid(-log(3)) = 0.25: a gate that starts a quarter open. The design contract
# asks for a "gentle" initialization in roughly 0.1-0.5; 0.25 sits in the middle,
# far from the 0 that would starve the proposal's gradient and from the 1 that
# would make the blend a no-op on the first step.
GATE_INITIAL_PROBABILITY = 0.25
ROLE_INITIAL_SCALE = 0.02
SOLVER_INITIAL_SCALE = 0.02


def step_encoding(step_index: int, dim: int, *, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
    """``[1, 1, dim]`` fixed harmonics of the internal step index *k*.

    Fixed rather than a learned table, for the same reason the positional read
    uses fixed harmonics instead of a learned query table: a table has to declare
    a maximum depth, and nothing else in this recurrence has one. The index is
    the *internal* step (the ``k`` axis of the design contract); it never encodes
    physical lead time, which is what ``LeadTimeEmbedding`` is for.
    """
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a nonnegative integer")
    dim = int(dim)
    if dim < 2 or dim % 2:
        raise ValueError("step encoding needs an even dim of at least 2")
    half = dim // 2
    frequencies = torch.exp(torch.linspace(
        0.0, -math.log(1000.0), half, device=device, dtype=torch.float32))
    angles = float(step_index) * frequencies * math.pi
    return torch.cat([angles.sin(), angles.cos()]).reshape(1, 1, dim).to(dtype)


class LocalSolverState(nn.Module):
    """One ConvGRU-style step of the per-patch working state ``Z``.

    The update mixes four sources per patch - the context ``C``, the encoded
    draft ``E(Y_k)``, the process read ``R_k`` and the step embedding - through a
    pointwise projection, one 3x3 depthwise convolution (the only spatial mixing,
    so the cost is O(N) with a 3x3 kernel and never a global N x N attention or a
    4D correlation volume), and a pointwise gate projection. The grid is the same
    patch grid the context uses; ``token_hw`` is what makes the 3x3 neighborhood
    the neighbourhood of the *image*, not of a flattened token order.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        dim = int(dim)
        if dim < 1:
            raise ValueError("solver state needs a positive dim")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) \
                or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.dim = dim
        self.kernel_size = kernel_size
        self.source_norm = nn.LayerNorm(3 * dim)
        self.project = nn.Linear(3 * dim, dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2,
                                   groups=dim, bias=False)
        self.state_norm = nn.LayerNorm(dim)
        self.gates = nn.Linear(dim, 3 * dim)

    def forward(self, z: torch.Tensor, *, context: torch.Tensor, draft_tokens: torch.Tensor,
                read: torch.Tensor, step_index: int,
                token_hw: tuple[int, int]) -> torch.Tensor:
        for name, value in (("z", z), ("context", context),
                            ("draft_tokens", draft_tokens), ("read", read)):
            if value.ndim != 3 or value.shape != z.shape:
                raise ValueError(f"{name} must be [B,N,D] and match the solver state exactly")
            if value.device != z.device:
                raise ValueError(f"{name} and the solver state are on different devices")
        rows, columns = int(token_hw[0]), int(token_hw[1])
        if rows < 1 or columns < 1 or rows * columns != z.shape[1]:
            raise ValueError(f"token_hw {tuple(token_hw)} does not match {z.shape[1]} positions")
        features = torch.cat([context, draft_tokens, read], dim=-1)
        hidden = self.project(self.source_norm(features))
        hidden = hidden + step_encoding(
            step_index, self.dim, device=z.device, dtype=z.dtype)
        grid = hidden.transpose(1, 2).reshape(z.shape[0], self.dim, rows, columns)
        grid = grid + self.depthwise(grid)
        gates = self.gates(self.state_norm(
            grid.flatten(2).transpose(1, 2)))
        update, reset, candidate = gates.chunk(3, dim=-1)
        update, reset = update.sigmoid(), reset.sigmoid()
        candidate = torch.tanh(candidate + reset * z)
        return (1.0 - update) * z + update * candidate


class PositionGate(nn.Module):
    """``g_k = sigmoid(.)`` as a per-position scalar on the patch grid.

    The bias starts the gate a quarter open, so at initialization the blend
    ``Y_k + g * (proposal - Y_k)`` already moves towards the proposal without
    drowning it: a gate initialized at 0 would hand the proposal no gradient at
    all, which is the failure the design contract calls out by name.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.score = nn.Linear(int(dim), 1)
        nn.init.normal_(self.score.weight, mean=0.0, std=1e-2)
        probability = float(GATE_INITIAL_PROBABILITY)
        nn.init.constant_(self.score.bias, math.log(probability / (1.0 - probability)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("the position gate reads [B,N,D] solver states")
        return self.score(self.norm(z)).sigmoid()


def expand_token_gate(gate: torch.Tensor, token_hw: tuple[int, int],
                      output_hw: tuple[int, int], patch_size: int) -> torch.Tensor:
    """``[B,N,1]`` per-token gate -> ``[B,1,H,W]`` on the native grid.

    Each token owns the same ``patch_size x patch_size`` block that
    ``CoarseForecastHead``'s transposed convolution decodes it into, so nearest
    replication followed by the decoder's own crop is the exact inverse mapping
    and not an interpolation. The gate therefore costs no parameter and cannot
    disagree with the decoder about which pixels a token covers.
    """
    if gate.ndim != 3 or gate.shape[-1] != 1:
        raise ValueError("the token gate must be [B,N,1]")
    rows, columns = int(token_hw[0]), int(token_hw[1])
    if rows < 1 or columns < 1 or gate.shape[1] != rows * columns:
        raise ValueError(f"token_hw {tuple(token_hw)} does not match {gate.shape[1]} gates")
    patch_size = int(patch_size)
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    grid = gate.transpose(1, 2).reshape(gate.shape[0], 1, rows, columns)
    grown = F.interpolate(grid, scale_factor=patch_size, mode="nearest")
    return crop_native_grid(grown, (int(output_hw[0]), int(output_hw[1])), patch_size)


def anchored_proposal(decoder: nn.Module, solver_state: torch.Tensor,
                      token_hw: tuple[int, int], output_hw: tuple[int, int],
                      anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``Y_proposal = X_t + Decoder(Z)``, returned with its tendency.

    ``anchor`` is the known state at the current physical time. Passing it here
    rather than re-deriving it from the history keeps the anchoring rule in one
    place; the alternative - recomputing "the last history step" at each call
    site - is how the three step implementations drifted apart in the first
    place.
    """
    if anchor.ndim != 4 or anchor.shape != (solver_state.shape[0], anchor.shape[1],
                                            output_hw[0], output_hw[1]):
        raise ValueError("the anchor must be [B,C_out,H,W] on the requested native grid")
    return decoder(solver_state, token_hw, output_hw, anchor)


def blend_forecast(draft: torch.Tensor, proposal: torch.Tensor,
                   gate: torch.Tensor) -> torch.Tensor:
    """``Y_(k+1) = Y_k + g_k * (Y_proposal - Y_k)`` on the native grid."""
    if draft.shape != proposal.shape or draft.ndim != 4:
        raise ValueError("draft and proposal must be equal [B,C,H,W] fields")
    expected = (draft.shape[0], 1, draft.shape[2], draft.shape[3])
    if gate.shape != expected:
        raise ValueError(f"the expanded gate must be {expected}, got {tuple(gate.shape)}")
    return draft + gate * (proposal - draft)
