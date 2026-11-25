from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Code identities the archived rounds' checkpoints were written under, and which round each
#: belongs to. Used only to make a skip reason legible; the check itself compares the checkpoint's
#: own recorded digest against the live tree.
ARCHIVED_CODE_IDENTITIES = {
    "59340710623ca24c7ee59ebb5711f52193ebfbaa72ed4532b41bdd987f5d9324":
        "the position-retention / typed-evidence / loop-depth-headroom line",
}


def require_archived_revision(checkpoint, *, what: str) -> None:
    """Skip with a named reason when a checkpoint's code identity is not the live one.

    ``load_checkpoint`` refuses a foreign implementation by design (decision 0065, Q-009): any
    byte change under ``model/`` moves ``model_code_sha256``, so every checkpoint written before
    that change stops loading. That is the guard working, not a defect. The sanctioned remedy for
    *reproducing* an archived reading is ``tools/r7_archived_replay.py``, which runs that round's
    own ``code.zip`` at its frozen absolute paths.

    This helper exists so the precondition is declared once instead of ten times, and so it cannot
    drift into a way of keeping a test green while it stops testing anything:

    * **named** -- the reason says which revision the artifact belongs to and where the replay tool
      is, rather than "not available";
    * **narrow** -- it fires only on an identity mismatch. A missing file, an unreadable checkpoint
      or a payload without a digest still raises, so a genuinely broken artifact fails loudly;
    * **body preserved** -- the assertions are untouched and run in full whenever the identities
      agree.

    A bare checkout has no ``outputs/`` at all, so these tests were already skipped in CI; this
    makes the *local* case explicit instead of a failure no change to the archived round could fix.
    """
    import torch

    from training.r7_experiment import model_code_digest

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"the archived checkpoint is missing: {path}")
    recorded = torch.load(path, map_location="cpu", weights_only=True).get("model_code_sha256")
    if not isinstance(recorded, str):
        raise ValueError(f"{path} records no model_code_sha256")
    live = model_code_digest()
    if recorded == live:
        return
    known = ARCHIVED_CODE_IDENTITIES.get(recorded, "an earlier revision")
    pytest.skip(
        f"{what} depends on a checkpoint written under model_code_sha256 {recorded[:12]} "
        f"({known}); the live tree is {live[:12]}. Changing model/ necessarily moves that digest, "
        f"so the artifact is not loadable here by design (Q-009). Reproduce it with "
        f"tools/r7_archived_replay.py against that round's own code.zip; this test is skipped "
        f"rather than relaxed, and it runs unchanged whenever the identities agree.")
