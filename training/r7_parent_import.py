"""Explicit M3 parent *weight* import, never archived execution or old resume.

The trust roots below are the independently accepted aux_off M3 artifacts. The
archive is read as bytes/AST metadata only. A fresh current-code Process model
must account for every parent tensor without adding or dropping a state key.
The original checkpoint, signed contract and outputs are never rewritten.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

import torch

from . import r7_experiment as experiment
from .r7_arm_harness import sha256_file

M3_PINS = {
    "protocol_sha256": "404cf32b8ee8f6c3ff192d46c1de6765abe4ae3fa72967469af800a774fde15d",
    "code_zip_sha256": "18595abce5acfa9e6f3252342f04eace470a06f48c3eea97c6ba463e3ea02d95",
    "model_code_sha256": "11090929930da4e1259698699cbbf12b3738cdfb2f609c3af516c24399144476",
    "sidecar_identity": "4fed1c78e4c4c09a41d95457649925b02d0c8ebf89aa47c2ac8dc6496734912d",
    "data_identity": "ef8c66911a70d6db222517e6a7e3f62bc32d2eef86efd4132e3bdd48266ccc07",
    "source_sha256": "496084a9260bacfaf6293a01d89439c1e49d6afa8f09bc1f51d89a1d1f9bda21",
}
M3_PARENTS = {
    41: {"checkpoint_sha256": "b486b41af65cd4266b3312d6b46c65a68328aa8198b6841d162b424b1ef9eea6",
         "signature": "93cdb77bdb5e671fb6443e89640c7ba1e9a4f293a85d2355d126c7755a6cdbce"},
    42: {"checkpoint_sha256": "70da2c5fdc62f0fdc33299b41d4367cd6d3e2c9984bc6fba423044aef4ae3471",
         "signature": "8aa879ff06643ead805afde71efeff171dc17d82362e9446c7c4c7598f4c3901"},
}
MAX_ARCHIVE_BYTES = 64 << 20
MAX_MEMBER_BYTES = 2 << 20
MAX_ARCHIVE_MEMBERS = 1024
CONFIG_MEMBERS = {"pyproject.toml", "requirements.txt", "requirements-r7-data.txt", "requirements-dev.txt"}
DIAGNOSTIC_WEIGHTS = ("input_diagnostic_weight", "future_diagnostic_weight", "draft_diagnostic_weight")


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _local_file(value, *, name=None):
    requested = Path(value)
    if "://" in str(value) or ".." in requested.parts:
        raise ValueError("local artifact paths only; unsafe path")
    if requested.name == "test.jsonl" or (name is not None and requested.name != name):
        raise ValueError(f"expected {name or 'non-test artifact'}; sealed test manifest forbidden")
    if any(part.is_symlink() for part in (requested, *requested.parents)):
        raise ValueError("symlink artifact paths are forbidden")
    path = requested.resolve()
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    return path


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact metadata must be a JSON object")
    experiment.canonical_digest(value)  # Reject NaN/Infinity even if JSON's reader accepts them.
    return value


def _member_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or str(path) != name or "\\" in name or ":" in name
            or "\x00" in name or any(part in (".", "..") or "legacy" in part for part in path.parts)):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if name not in CONFIG_MEMBERS and (path.parts[0] not in ("model", "training", "data", "scripts")
                                       or path.suffix != ".py"):
        raise ValueError(f"unsupported archive member: {name}")
    return path


def _constructor_metadata(content):
    """Literal constructor metadata, not an interpreter for archived code."""
    tree = ast.parse(content.decode("utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "ProcessForecastCoReasoner"]
    if len(classes) != 1:
        raise ValueError("archive lacks unambiguous Process constructor metadata")
    constructors = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
                    and node.name == "__init__"]
    if len(constructors) != 1:
        raise ValueError("archive lacks unambiguous Process constructor")
    arguments = constructors[0].args
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise ValueError("variadic archived constructor cannot prove exact configuration")
    positional = arguments.posonlyargs + arguments.args
    names = [arg.arg for arg in positional + arguments.kwonlyargs if arg.arg != "self"]
    defaults = dict(zip((arg.arg for arg in positional[-len(arguments.defaults):]),
                        (ast.literal_eval(node) for node in arguments.defaults)))
    defaults.update({arg.arg: ast.literal_eval(node) for arg, node in
                     zip(arguments.kwonlyargs, arguments.kw_defaults) if node is not None})
    if defaults.get("local_solver_state") is not False or defaults.get("default_reasoning_steps") != 4:
        raise ValueError("archived Process must default to RW-A local_solver_state=False and K=4")
    return {"class": "ProcessForecastCoReasoner", "arguments": names, "defaults": defaults}


def _verify_archive(path, protocol):
    code = protocol["code"]
    content = path.read_bytes()
    if len(content) > MAX_ARCHIVE_BYTES or _sha(content) != M3_PINS["code_zip_sha256"]:
        raise ValueError("parent code zip digest/size mismatch")
    if code["code_zip_sha256"] != _sha(content):
        raise ValueError("protocol code zip digest mismatch")
    expected = code["files"]
    for name in expected:
        _member_path(name)
    actual, model_contents = {}, {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("invalid archive member count")
        total = 0
        for member in members:
            _member_path(member.filename)
            total += member.file_size
            if (member.filename in actual or stat.S_ISLNK(member.external_attr >> 16)
                    or member.flag_bits & 1 or member.file_size > MAX_MEMBER_BYTES
                    or total > MAX_ARCHIVE_BYTES):
                raise ValueError("duplicate, symlink, encrypted or oversized archive member")
            payload = archive.read(member)
            actual[member.filename] = _sha(payload)
            if member.filename.startswith("model/"):
                model_contents[member.filename.removeprefix("model/")] = payload
    if actual != expected or experiment.canonical_digest(actual) != code["source_tree_sha256"]:
        raise ValueError("archive member hash map mismatch (missing/extra/tampered source)")
    if "__init__.py" not in model_contents or "process_forecast_r7.py" not in model_contents:
        raise ValueError("archive is missing Process/model sources")
    digest = hashlib.sha256()
    for name, payload in sorted(model_contents.items()):
        digest.update(name.encode() + b"\0")
        digest.update(payload)
    model_digest = digest.hexdigest()
    if model_digest != code["model_code_sha256"] or model_digest != M3_PINS["model_code_sha256"]:
        raise ValueError("archived model source bytes digest mismatch")
    return {"code_zip_sha256": _sha(content), "files": actual,
            "source_tree_sha256": experiment.canonical_digest(actual),
            "model_code_sha256": model_digest,
            "model_files": {name: _sha(payload) for name, payload in sorted(model_contents.items())},
            "constructor": _constructor_metadata(model_contents["process_forecast_r7.py"]),
            "execution": "bytes and AST metadata only; no extraction/exec/import"}


def verify_parent_checkpoint_signature(checkpoint, *, archived_model_sha256):
    """Verify the unchanged old contract and *old* model digest, not today's code.

    M3 stores its implementation digest at checkpoint top level, outside the
    signed training contract. Both identities are checked independently; no
    current digest is injected into the old contract or saved payload.
    """
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "r7-local-v1":
        raise ValueError("unsupported parent checkpoint format")
    contract = checkpoint.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("missing parent checkpoint contract")
    if (checkpoint.get("model_code_sha256") != archived_model_sha256
            or ("model_code_sha256" in contract and contract["model_code_sha256"] != archived_model_sha256)):
        raise ValueError("parent checkpoint model digest differs from archived source bytes")
    signature = experiment.canonical_digest(contract)
    if checkpoint.get("signature") != signature:
        raise ValueError("parent checkpoint contract signature digest mismatch")
    return signature


def _verify_contract(saved, checkpoint_hash, protocol):
    contract = saved["contract"]
    seed = contract.get("seed")
    if type(seed) is not int or seed not in M3_PARENTS:
        raise ValueError("only the two explicitly accepted M3 aux_off parents are supported")
    pins = M3_PARENTS[seed]
    if checkpoint_hash != pins["checkpoint_sha256"] or saved["signature"] != pins["signature"]:
        raise ValueError("accepted parent checkpoint SHA256/signature mismatch")
    arms = [arm for arm in protocol["arms"] if arm["name"] == "aux_off"]
    if len(arms) != 1:
        raise ValueError("original protocol must contain one aux_off arm")
    arm = arms[0]
    from .r7_m3_protocol import expected_training_contract
    if (contract["kind"] != "process" or arm["kind"] != "process"
            or contract["model"] != arm["model_config"] or contract["steps"] != 4
            or contract["model"].get("default_reasoning_steps") != 4
            or contract["model"].get("local_solver_state", False) is not False
            or contract["data_identity"] != M3_PINS["data_identity"]
            or contract.get("process_weight") != 0 or type(saved.get("updates")) is not int
            or saved["updates"] != 400 or protocol["shared_controls"]["updates"] != 400
            or contract.get("process_supervision") != expected_training_contract(protocol, "aux_off")
            or any(arm["supervision_weights"][key] != 0 for key in DIAGNOSTIC_WEIGHTS)):
        raise ValueError("parent signed M3 aux_off architecture/fixed inverse/training contract differs")


def _verify_data(protocol, contract, sidecar, trainmanifest):
    from data.preprocess.r7_process_scale_sidecar import load_process_scale_sidecar
    from .r7_m3_identity import source_identity
    from .r7_process_training_contract import _root_inverse
    metadata = load_process_scale_sidecar(sidecar)
    pins, data = protocol["sidecar"], protocol["data"]
    if (str(sidecar) != pins["path"] or str(trainmanifest.parent) != protocol["manifests"]
            or str(trainmanifest) != metadata["train_manifest"]
            or metadata["sidecar_identity"] != M3_PINS["sidecar_identity"]
            or metadata["sidecar_identity"] != pins["identity"]
            or sha256_file(sidecar) != pins["metadata_sha256"]
            or metadata["preflight_identity"] != pins["preflight_identity"]
            or metadata["data_identity"] != data["data_identity"]
            or metadata["store"] != data["store"]
            or sha256_file(_local_file(sidecar.parent / "BUILD_COMPLETE.json")) != pins["publication_marker_sha256"]):
        raise ValueError("parent sidecar/path/data/publication identity mismatch")
    identity, reader = experiment.dataset_identity(trainmanifest)
    if identity != M3_PINS["data_identity"] or identity != contract["data_identity"]:
        raise ValueError("actual train data/normalization identity mismatch")
    if (len(reader) != contract["dataset_length"] or len(reader) != data["train_windows"]
            or any(record["split"] != "train" or len(record["history_indices"]) != 2
                   or record["lead_time_hours"] != 6 for record in reader.records)
            or [record["sample_id"] for record in reader.records] != metadata["train_sample_ids"]
            or {(reader.manifest.parent / record["store_path"]).resolve() for record in reader.records}
            != {Path(metadata["store"])}):
        raise ValueError("actual train manifest/store/sample ownership mismatch")
    root = reader._store(reader.records[0])
    fixed = _root_inverse(root, identity)
    supervision = contract["process_supervision"]
    if (fixed != supervision["fixed_train_context"]
            or experiment.canonical_digest(fixed) != supervision["fixed_train_context_sha256"]
            or list(root["state"].shape) != data["shape"]):
        raise ValueError("actual fixed train inverse/channel order/shape mismatch")
    sources = source_identity(trainmanifest.parent, root)
    source = metadata["source_identity"]
    if (sources != protocol["sources"] or sources["source_sha256"] != M3_PINS["source_sha256"]
            or source["sha256"] != sources["source_sha256"] or source["path"] != sources["source_path"]
            or source["bytes"] != sources["source_bytes"]
            or source["source_preflight_sha256"] != sources["preflight_report_sha256"]
            or metadata["train_manifest_sha256"] != sources["train_manifest_sha256"]):
        raise ValueError("parent source/preflight/receipt/BUILD_COMPLETE/manifest identity mismatch")
    return {"data_identity": identity, "train_manifest": str(trainmanifest),
            "train_manifest_sha256": sources["train_manifest_sha256"], "sources": sources,
            "sidecar_identity": metadata["sidecar_identity"], "sidecar_metadata_sha256": sha256_file(sidecar),
            "fixed_train_context": fixed, "fixed_train_context_sha256": experiment.canonical_digest(fixed),
            "diagnostic_active_mask": metadata["active_mask"],
            "identity_scope": "store metadata/norms and opaque source bytes; no weather/test fields read"}


def _effective_config(spec, constructor):
    if not isinstance(spec, dict) or spec.get("architecture", "window") != "window":
        raise ValueError("Process exact import requires a window model specification")
    options = {key: value for key, value in spec.items() if key != "architecture"}
    names, defaults = set(constructor["arguments"]), constructor["defaults"]
    if set(options) - names or names - set(defaults) - set(options):
        raise ValueError("model specification has unknown/missing archived constructor arguments")
    return {**defaults, **options}


def tensor_sha256(tensor):
    """Hash dtype, shape and exact contiguous CPU bytes (also supports BF16)."""
    value = tensor.detach().cpu().contiguous()
    header = {"shape": list(value.shape), "dtype": str(value.dtype)}
    digest = hashlib.sha256(experiment.canonical_digest(header).encode() + b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _copy_exact(model, state):
    target = model.state_dict()
    if not isinstance(state, dict) or not state or set(state) != set(target):
        missing = sorted(set(target) - set(state or {}))
        extra = sorted(set(state or {}) - set(target))
        raise ValueError(f"exact parent state keys mismatch; missing={missing}, unused/extra={extra}")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.dtype != target[name].dtype or tuple(tensor.shape) != tuple(target[name].shape)):
            raise ValueError(f"parent tensor shape/dtype/layout mismatch: {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"nonfinite parent tensor: {name}")
    # load_state_dict copies, not assigns, so there is no shared parent storage.
    model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    parameters = dict(model.named_parameters())
    mapping = []
    for name in sorted(state):
        previous, current = tensor_sha256(state[name]), tensor_sha256(actual[name])
        if previous != current:
            raise ValueError(f"parent weight copy changed tensor bytes: {name}")
        mapping.append({"source_key": name, "target_key": name, "shape": list(state[name].shape),
                        "dtype": str(state[name].dtype), "old_sha256": previous, "new_sha256": current,
                        "kind": "parameter" if name in parameters else "buffer"})
    model.zero_grad(set_to_none=True)
    return mapping


def import_parent(checkpoint, *, original_protocol, codezip, sidecar, trainmanifest,
                  target_kind="process", model_spec=None):
    """Return ``(current_code_model, JSON_provenance_report)`` with copied weights.

    Only process->process exact-state import is implemented. Generic requires an
    owner-defined complete mapping and is rejected, not partially initialized.
    Inputs are read-only. No optimizer, scheduler, RNG, epoch/cursor or update
    state is resumed. The child runner must create its own training protocol and
    contract, retaining this report as separate parent provenance.
    """
    if target_kind != "process":
        raise ValueError("only Process exact-key import is supported; Generic requires owner mapping")
    paths = {"checkpoint": _local_file(checkpoint), "original_protocol": _local_file(original_protocol),
             "codezip": _local_file(codezip), "sidecar": _local_file(sidecar, name="scale_metadata.json"),
             "trainmanifest": _local_file(trainmanifest, name="train.jsonl")}
    protocol = _json(paths["original_protocol"])
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if (experiment.canonical_digest(body) != protocol.get("protocol_sha256")
            or protocol["protocol_sha256"] != M3_PINS["protocol_sha256"]
            or protocol.get("format") != "r7-73-process-supervision-protocol-v1"
            or protocol.get("scientific_claim") is not False or protocol.get("test_read") is not False
            or protocol.get("frozen_before_any_step") is not True):
        raise ValueError("original frozen protocol canonical digest/acceptance mismatch")
    archive = _verify_archive(paths["codezip"], protocol)
    content = paths["checkpoint"].read_bytes()
    checkpoint_hash = _sha(content)
    if checkpoint_hash not in {pins["checkpoint_sha256"] for pins in M3_PARENTS.values()}:
        raise ValueError("accepted parent checkpoint SHA256 mismatch before deserialization")
    saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    verify_parent_checkpoint_signature(saved, archived_model_sha256=archive["model_code_sha256"])
    _verify_contract(saved, checkpoint_hash, protocol)
    data = _verify_data(protocol, saved["contract"], paths["sidecar"], paths["trainmanifest"])
    spec = deepcopy(saved["contract"]["model"] if model_spec is None else model_spec)
    previous_config = _effective_config(saved["contract"]["model"], archive["constructor"])
    target_config = _effective_config(spec, archive["constructor"])
    if type(target_config.get("detach_between_steps")) is not bool:
        raise ValueError("detach_between_steps must be boolean")
    changes = {key: {"parent": previous_config[key], "target": value}
               for key, value in target_config.items() if value != previous_config[key]}
    if set(changes) - {"detach_between_steps"}:
        raise ValueError("target model specification changes the accepted parent architecture")
    current_digest = experiment.model_code_digest()
    model = experiment.make_model(target_kind, spec)
    mapping = _copy_exact(model, saved["model"])
    if experiment.model_code_digest() != current_digest:
        raise ValueError("current model source changed while constructing parent import")
    initialization = {"format": "r7-parent-initialization-v1", "kind": target_kind, "model": spec,
                      "data_identity": data["data_identity"], "model_code_sha256": current_digest,
                      "operation": "explicit-exact-state-weight-copy; not resume"}
    report = {
        "format": "r7-parent-weight-import-v1", "scientific_claim": False,
        "limitations": ["identity/weight-copy acceptance only; no scientific or forecast-equivalence claim",
                        "legacy no-calendar numeric equivalence must be measured separately",
                        "source is hashed opaquely; data identity does not hash every atmospheric chunk",
                        "Generic transfer is unsupported until an explicit owner mapping exists"],
        "parent": {"checkpoint": str(paths["checkpoint"]), "checkpoint_sha256": checkpoint_hash,
                   "signature": saved["signature"], "contract": deepcopy(saved["contract"]),
                   "model_code_sha256": saved["model_code_sha256"], "updates": saved["updates"],
                   "protocol": str(paths["original_protocol"]), "protocol_sha256": protocol["protocol_sha256"]},
        "archive": archive, "data": data, "mapping": mapping,
        "unused_source_keys": [], "uninitialized_target_keys": [],
        "parent_state_sha256": experiment.canonical_digest({entry["source_key"]: entry["old_sha256"] for entry in mapping}),
        "target_state_sha256": experiment.canonical_digest({entry["target_key"]: entry["new_sha256"] for entry in mapping}),
        "initialization_contract": initialization, "initialization_signature": experiment.canonical_digest(initialization),
        "declared_behavior_changes": changes,
        "gradient_semantics": "detach_between_steps may change internal-K gradient flow; never an old training resume",
        "optimizer_reset": True, "rng_restored": False, "resume": False,
        "child_counters": {"updates": 0, "epoch": 0, "cursor": 0},
        "special_diagnostics": {
            "process_readout": "all keys copied; scalar diagnostic head is not a forecast input; parent aux_off weights are zero",
            "process_to_context": "all keys copied; pooled fallback inactive when positional_process_readout=True",
            "inactive_scale_channels": [index for index, active in enumerate(data["diagnostic_active_mask"]) if not active],
        },
    }
    report["report_sha256"] = experiment.canonical_digest(report)
    return model, report
