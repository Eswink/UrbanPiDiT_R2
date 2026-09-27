"""Structural equivalence between the B2/B3 ``generic`` and ``process`` arms.

Background (#65 pre-diagnostic finding). B2 and B3 declare the recursive pair
as the *shared-structure* comparison, with ``generic`` as the shared
initialization anchor. The harness copies the anchor's ``state_dict`` entries
into ``process`` by exact name. Two facts make that pair weaker than the
declaration suggests:

1. ``generic`` stores its recurrent cell and latent under ``cell.*`` / ``latent``
   / ``latent_to_context.*``, while ``process`` uses ``reasoning_cell.*`` /
   ``process_queries`` / ``process_to_context.*``. The name-based copy therefore
   skips the whole recurrent cell (39 of 111 tensors), and the two arms do keep
   the seeded initialization there - they are both seeded from the same stream,
   so those tensors agree bit-for-bit. What the copy does *not* align is the
   final ``latent_to_context``/``process_to_context`` projection: it is built
   from the seeded stream at a different point, so the two differ.

2. The auxiliary process loss was run at weight 0 in both B2 and B3. The
   process readout is off the forecast path, so at weight 0 it receives no
   gradient at all: the process arm is a generic recursive reasoner whose
   recurrent state is renamed, with a dead readout head attached.

The tests below pin the consequence, which is what an attribution argument has
to rest on: once the single non-aligned projection is forced equal, the two
models are the *same function* - identical forecasts, drafts and recurrent
states. That is a property of the architecture and the harness, not of any
dataset, so it is checked on tiny synthetic tensors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.r7_halting import forecast_inputs  # noqa: E402
from training.r7_experiment import make_model, seed_everything  # noqa: E402

TINY = {"in_channels": 3, "out_channels": 3, "history_steps": 2,
        "architecture": "window", "dim": 16, "depth": 1, "heads": 2,
        "window_size": 2, "patch_size": 2, "dropout": 0.0,
        "default_reasoning_steps": 2}


def _configs():
    generic = dict(TINY, latent_tokens=3)
    process = dict(TINY, anchored_processes=2, free_processes=1,
                   use_forecast_feedback=True)
    return generic, process


def _gaussian_name_map(name):
    if name.startswith("cell."):
        return "reasoning_cell." + name[5:]
    if name == "latent":
        return "process_queries"
    if name.startswith("latent_to_context."):
        return "process_to_context." + name[len("latent_to_context."):]
    return name


def _arm_pair(seed=3):
    """Build both arms exactly the way the B2/B3 harness does."""
    generic_config, process_config = _configs()
    seed_everything(seed)
    anchor = make_model("generic", generic_config)
    anchor_state = {name: tensor.clone()
                    for name, tensor in anchor.state_dict().items()}
    seed_everything(seed)
    process = make_model("process", process_config)
    target = process.state_dict()
    transfer = {name: tensor for name, tensor in anchor_state.items()
                if name in target and tuple(target[name].shape) == tuple(tensor.shape)}
    process.load_state_dict(transfer, strict=False)
    return anchor_state, process


def _batch():
    generator = torch.Generator().manual_seed(0)
    return {"coarse_history": torch.randn(2, 2, 3, 4, 4, generator=generator),
            "latitude": torch.tensor([[30.0, 31.0, 32.0, 33.0]] * 2)}


def test_name_based_copy_skips_the_recurrent_cell():
    """The harness's transfer list is exactly the non-recursive prefix."""
    anchor_state, process = _arm_pair()
    target = process.state_dict()
    transfer = {name for name in anchor_state
                if name in target
                and tuple(target[name].shape) == tuple(anchor_state[name].shape)}
    assert not any(name.startswith("cell.") for name in transfer)
    assert "latent" not in transfer
    assert not any(name.startswith("latent_to_context.") for name in transfer)
    assert any(name.startswith("backbone.") for name in transfer)


def test_the_two_arms_differ_only_in_the_final_projection():
    anchor_state, process = _arm_pair()
    process_state = process.state_dict()
    mapped = [(name, _gaussian_name_map(name)) for name in anchor_state
              if _gaussian_name_map(name) in process_state]
    assert len(mapped) == len(anchor_state)
    differing = [pair for pair in mapped
                 if not torch.equal(anchor_state[pair[0]], process_state[pair[1]])]
    assert differing == [("latent_to_context.1.weight", "process_to_context.1.weight"),
                         ("latent_to_context.1.bias", "process_to_context.1.bias")]


def test_the_process_readout_is_off_the_forecast_path():
    """At auxiliary weight 0 the readout head is dead, not merely untrained."""
    anchor_state, process = _arm_pair()
    process.train()
    from training.r7_recursive_losses import deep_supervised_forecast_mse
    out = process(forecast_inputs(_batch()), reasoning_steps=2)
    loss = deep_supervised_forecast_mse(out.draft_forecasts,
                                        torch.zeros(2, 3, 4, 4))
    process.zero_grad(set_to_none=True)
    loss.backward()
    for name, parameter in process.named_parameters():
        if name.startswith("process_readout."):
            assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
               for name, parameter in process.named_parameters()
               if not name.startswith("process_readout."))


def test_aligning_the_projection_makes_the_arms_the_same_function():
    """The headline consequence: generic and process are one function modulo
    the single non-aligned projection and the dead readout."""
    generic_config, process_config = _configs()
    anchor_state, process = _arm_pair()
    forced = process.state_dict()
    forced["process_to_context.1.weight"] = anchor_state["latent_to_context.1.weight"].clone()
    forced["process_to_context.1.bias"] = anchor_state["latent_to_context.1.bias"].clone()
    process.load_state_dict(forced, strict=True)
    generic = make_model("generic", generic_config)
    generic.load_state_dict(anchor_state, strict=True)
    generic.eval()
    process.eval()
    batch = _batch()
    with torch.no_grad():
        generic_out = generic(forecast_inputs(batch))
        process_out = process(forecast_inputs(batch), reasoning_steps=2)
    assert torch.equal(generic_out.forecast, process_out.forecast)
    assert torch.equal(generic_out.draft_forecasts, process_out.draft_forecasts)
    assert torch.equal(generic_out.initial_forecast, process_out.initial_forecast)
    assert torch.equal(generic_out.latent_state, process_out.process_state)


def test_the_equivalence_is_not_an_artefact_of_a_single_seed():
    for seed in (1, 2, 3, 41):
        generic_config, _ = _configs()
        anchor_state, process = _arm_pair(seed=seed)
        forced = process.state_dict()
        forced["process_to_context.1.weight"] = anchor_state["latent_to_context.1.weight"].clone()
        forced["process_to_context.1.bias"] = anchor_state["latent_to_context.1.bias"].clone()
        process.load_state_dict(forced, strict=True)
        generic = make_model("generic", generic_config)
        generic.load_state_dict(anchor_state, strict=True)
        generic.eval()
        process.eval()
        with torch.no_grad():
            generic_out = generic(forecast_inputs(_batch()))
            process_out = process(forecast_inputs(_batch()), reasoning_steps=2)
        assert torch.equal(generic_out.forecast, process_out.forecast), seed


def test_the_equivalence_does_not_hold_at_auxiliary_weight_zero_alone():
    """Counterproof: the *unaligned* pair is genuinely two different functions,
    so the equivalence above is a statement about the projection, not a vacuous
    identity between two identical models."""
    anchor_state, process = _arm_pair()
    generic_config, _ = _configs()
    generic = make_model("generic", generic_config)
    generic.load_state_dict(anchor_state, strict=True)
    generic.eval()
    process.eval()
    batch = _batch()
    with torch.no_grad():
        generic_out = generic(forecast_inputs(batch))
        process_out = process(forecast_inputs(batch), reasoning_steps=2)
    assert not torch.equal(generic_out.forecast, process_out.forecast)
    assert float((generic_out.forecast - process_out.forecast).abs().max()) > 0.0
