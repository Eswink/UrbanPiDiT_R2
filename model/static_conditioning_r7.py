"""Time-invariant urban static conditioning for the main forecaster (#S3 wiring round).

Decision 0063 froze *how* the urban static covariates are produced and published, and
decision 0069 authorises wiring them into the model. Both say the same thing about identity:
the static field's ``static_identity`` is bound into the experiment's ``protocol_sha256``,
**never** into ``data_identity``. Writing the fields into the zarr store's attributes would
change ``data_identity`` and decouple every registered R1-R5 receipt; leaving them unregistered
would let two different static fields share one ``data_identity``. Neither is acceptable, so
the field stays a separate published product and the model reads it by identity.

What this module is, and what it deliberately is not:

- **A time-invariant additive conditioning term.** ``static_conditioning.py`` maps the static
  field to one vector per patch token and *adds* it to the encoder context, exactly the shape
  of the existing ``SpacetimeConditioning`` / ``KnownContextConditioning`` terms. Static
  means static: the term is a function of position only, so the same field is added at every
  lead time and every reasoning step. It never reads a target, a future observation or a clock.
- **Not an extra state channel.** No store, no ``attrs``, no ``data_identity``. The field is
  loaded from the published ``static_fields.npz`` and its identity is checked against the
  published manifest before a single number is used.
- **No new parameters, so no capacity confound.** The feature encoder is a fixed, declared
  polynomial expansion rather than a learned MLP. Decision 0069 clause 4 requires a declared
  ``parameter_delta``; here it is exactly zero, which means an ``on``/``off`` difference
  cannot be attributed to added capacity. The trade-off is stated in the round's evidence
  page: a fixed expansion cannot learn which covariate matters, so a null is *not* evidence
  that the covariates are uninformative.

The off path is the pre-existing computation, operation for operation: with
``static_conditioning=False`` nothing is constructed, nothing is read, and no tensor enters
the forward pass. Decision 0069's exit W2 pins that as a bitwise, pre-/post-edit comparison
rather than an assertion here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .layers.patch_grid import pad_patch_grid

#: Repository-relative, resolved against the checkout that owns this package rather than the
#: process working directory: a driver run from another cwd must not silently resolve the
#: product against a different tree.
REPO_ROOT = Path(__file__).resolve().parents[1]
#: The published product this module reads. Recorded, not discovered: a round that silently
#: picked up a different attempt would be an unregistered experiment change.
STATIC_MANIFEST = REPO_ROOT / "outputs/r7_urban_static_covariates_attempt01/published/manifest/manifest.json"
STATIC_FIELDS = REPO_ROOT / "outputs/r7_urban_static_covariates_attempt01/published/fields/static_fields.npz"
#: Decision 0063's published identity, pinned here so a swapped product raises instead of
#: being conditioned on. It is one of the fields bound into ``protocol_sha256``.
STATIC_IDENTITY = "8671ed7ce0c82d1ec9ac834e68e2e3c7bbc30e14fa6b8a07501e070d27708b03"

#: The default covariate set: one value per variable, time-invariant by construction.
#: ``*_frac_2021`` / ``*_2020`` are the newest epoch of each multi-epoch layer; using a single
#: epoch keeps the term time-invariant, which is what makes it a *static* conditioner. The
#: multi-epoch arrays stay in the product for a future time-varying round.
DEFAULT_VARIABLES = (
    "lc_built_frac_2021", "lc_tree_frac_2021", "lc_crop_frac_2021",
    "lc_water_frac_2021", "lc_bare_frac_2021",
    "built_surface_frac_2020", "building_height_m", "pop_density_km2_2020",
)

#: Scale each covariate to a comparable magnitude before the expansion. Fractions are already
#: in [0, 1]; the two dimensional variables are divided by their published maxima
#: (``building_height_m`` max 16.201997756958008, ``pop_density_km2_2020`` max 22490.138671875)
#: so no single covariate dominates the sum by three orders of magnitude. These are the
#: published maxima rounded to nine significant digits, not fitted statistics.
STATIC_SCALES = {
    "building_height_m": 16.2019978, "pop_density_km2_2020": 22490.1387,
}
#: Fixed polynomial expansion: per covariate, its centred value and its square. The square
#: term is what lets a monotone covariate act non-monotonically without a learned layer.
EXPANSION_PER_VARIABLE = 2


def require_static_identity(manifest_path: Path = STATIC_MANIFEST,
                            fields_path: Path = STATIC_FIELDS) -> dict:
    """Return the manifest only if it is complete and matches the pinned identity.

    Failure is explicit by design, and for the same reason ``require_spacetime_fields``
    raises: the alternatives are substituting a placeholder field or conditioning on a
    product nobody registered. There is no default branch here to fall into.
    """
    manifest_path, fields_path = Path(manifest_path), Path(fields_path)
    complete = manifest_path.parent / "BUILD_COMPLETE.json"
    if not complete.is_file():
        raise FileNotFoundError(f"static product is not published: {complete} is missing")
    if not json.loads(complete.read_text(encoding="utf-8")).get("build_complete"):
        raise RuntimeError(f"static product is incomplete according to {complete}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = manifest.get("static_identity")
    if observed != STATIC_IDENTITY:
        raise RuntimeError(f"static identity {observed} is not the pinned {STATIC_IDENTITY}; "
                           "a different field must be a different protocol, not a silent swap")
    if not fields_path.is_file():
        raise FileNotFoundError(f"published static field is missing: {fields_path}")
    return manifest


def load_static_fields(variables=DEFAULT_VARIABLES, *, manifest_path: Path = STATIC_MANIFEST,
                       fields_path: Path = STATIC_FIELDS) -> tuple[torch.Tensor, list[str]]:
    """``[K, H, W]`` float32 stack of the declared variables, in the declared order.

    The product's own row order is preserved: row 0 is the northernmost cell, column 0 the
    westernmost, so the stack lines up with ``latitude`` descending / ``longitude`` ascending
    exactly as the store's grid does. A missing variable raises rather than being skipped,
    because a silently shorter stack would be a different conditioner under the same protocol.
    """
    require_static_identity(manifest_path, fields_path)
    with torch.no_grad():
        import numpy as np
        archive = np.load(fields_path)
        missing = [name for name in variables if name not in archive.files]
        if missing:
            raise KeyError(f"published static field lacks {missing}")
        stack = torch.stack([
            torch.from_numpy(np.asarray(archive[name], dtype=np.float32)) for name in variables])
    if stack.ndim != 3:
        raise ValueError(f"each static variable must be [H, W], got {tuple(stack.shape)}")
    if not torch.isfinite(stack).all():
        raise ValueError("static field contains a non-finite value")
    return stack, list(variables)


def static_features(field: torch.Tensor, *, token_hw: tuple[int, int], patch_size: int,
                    periodic_width: bool = False, variables=DEFAULT_VARIABLES) -> torch.Tensor:
    """``[N, 2K]`` patch-averaged, scaled, centred covariates and their squares.

    Patch averaging uses the same padding the encoder applies (``pad_patch_grid``), so token
    *i* describes the same cell the encoder produced it from - the invariant
    ``position_features`` relies on. Centring is per covariate over the region: the mean and
    the half-range come from *this* field, so the expansion cannot depend on another product.
    """
    count = field.shape[0]
    if count != len(variables):
        raise ValueError(f"the field carries {count} variables, the declared set {len(variables)}")
    rows, columns = field.shape[-2:]
    scales = torch.tensor([STATIC_SCALES.get(name, 1.) for name in variables],
                          dtype=field.dtype, device=field.device).reshape(-1, 1, 1)
    scaled = field / scales
    padded = pad_patch_grid(scaled.unsqueeze(0), patch_size, periodic_width=periodic_width)
    token_rows, token_columns = token_hw
    if tuple(padded.shape[-2:]) != (token_rows * patch_size, token_columns * patch_size):
        raise ValueError("padded static grid does not match the encoder token grid")
    cells = padded.reshape(1, count, token_rows, patch_size, token_columns,
                           patch_size).mean(dim=(3, 5))
    flat = cells.reshape(count, -1)
    centre = flat.mean(dim=1, keepdim=True)
    half_range = (flat.amax(dim=1, keepdim=True) - flat.amin(dim=1, keepdim=True)).clamp(min=1e-6)
    centred = ((flat - centre) / (0.5 * half_range)).clamp(-3., 3.)
    return torch.cat([centred, centred.square()], dim=0).transpose(0, 1).contiguous()


class StaticConditioning(nn.Module):
    """Additive, parameter-free, time-invariant per-token static term (decision 0069).

    The module owns the field as a buffer, not a parameter: it must not be trained, and it
    must be identical for every seed, lead time and reasoning step - that is what "static"
    means here. ``parameter_delta`` is therefore exactly zero and is asserted by the round's
    protocol rather than estimated.
    """

    def __init__(self, field: torch.Tensor, *, dim: int, patch_size: int,
                 periodic_width: bool = False, variables=DEFAULT_VARIABLES, hidden: int = 128):
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.periodic_width = bool(periodic_width)
        self.variables = tuple(variables)
        if self.dim < 1 or self.patch_size < 1:
            raise ValueError("dim and patch_size must be positive")
        features = EXPANSION_PER_VARIABLE * len(self.variables)
        if hidden < 1:
            raise ValueError("hidden width must be positive")
        # Deterministic projection: a fixed, seeded tensor rather than a learned layer keeps
        # the parameter delta at zero. It is a buffer so it travels with the checkpoint and
        # cannot be retrained by accident.
        generator = torch.Generator().manual_seed(0)
        projection = torch.randn(features, self.dim, generator=generator) / features ** 0.5
        self.register_buffer("field", field.detach().to(torch.float32), persistent=True)
        self.register_buffer("projection", projection.to(torch.float32), persistent=True)

    def forward(self, *, token_hw: tuple[int, int]) -> torch.Tensor:
        """``[1, N, dim]`` additive term; batch-invariant, so it is not expanded here."""
        features = static_features(self.field, token_hw=token_hw, patch_size=self.patch_size,
                                   periodic_width=self.periodic_width, variables=self.variables)
        return (features.to(self.projection.dtype) @ self.projection).unsqueeze(0)


def require_static_batch_fields(batch: Mapping[str, torch.Tensor]) -> None:
    """Reject a batch that wants static conditioning where the pathway cannot read it.

    A round that asks for the conditioner and forgets to declare its product would otherwise
    get the module's own buffer (one fixed field) while believing it acted on the sample's.
    There is exactly one published product, so there is nothing per-sample to check - the
    check is that the caller has not asked for a mode this module cannot honour.
    """
    mode = batch.get("static_conditioning_mode")
    if mode is not None and mode != "published":
        raise ValueError(f"static_conditioning_mode must be 'published', got {mode!r}")
