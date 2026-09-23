"""V5.3 compatibility wrapper for the V5.3.1 reviewer-evidence harness."""

from __future__ import annotations

try:
    from .run_v531_reviewer_evidence_checks import main
except ImportError:  # pragma: no cover
    from run_v531_reviewer_evidence_checks import main


if __name__ == "__main__":
    main()
