"""UrbanPiDiT V5.3.1 reviewer-evidence version authority.

该模块是 V5.3.1 审稿证据版本的单一命名来源。目录名保持历史工程约定，
论文、脚本、发布包和诊断输出统一使用这里的 release metadata。
"""

from __future__ import annotations

from dataclasses import dataclass


VERSION = "5.3.1"
VERSION_SHORT = "V5.3.1"
VERSION_NAME = "UrbanPiDiT-V5.3.1-MorphoProcessDiT"
METHOD_FAMILY = "Morphology-conditioned Process Diffusion Transformer"
RELEASE_NAME = "UrbanPiDiT-V5.3.1-MorphoProcessDiT-ReviewerEvidence"
RELEASE_SLUG = "urbanpidit_v531_morphoprocessdit_reviewer_evidence"
PACKAGE_DIRECTORY_NAME = "UrbanPiDiT_V4_with_baselines"
DEFAULT_RUN_NAME = "urbanpidit_v531_morphoprocessdit"

CLAIM_BOUNDARY = (
    "morphology-derived process proxies are diagnostic and conditioning signals; "
    "they are not claimed as measured physical parameters or causal proof."
)

DIAGNOSTIC_PREFIXES = (
    "version/",
    "model/",
    "input/",
    "output/",
    "proxy/",
    "process_adaln/",
    "urban_control/",
    "process_graph/",
    "micromet/",
    "residual_prediction/",
    "leakage/",
)


@dataclass(frozen=True)
class UrbanPiDiTVersionInfo:
    """Reviewer-facing version metadata."""

    version: str = VERSION
    version_short: str = VERSION_SHORT
    version_name: str = VERSION_NAME
    method_family: str = METHOD_FAMILY
    release_name: str = RELEASE_NAME
    release_slug: str = RELEASE_SLUG
    package_directory_name: str = PACKAGE_DIRECTORY_NAME
    default_run_name: str = DEFAULT_RUN_NAME
    claim_boundary: str = CLAIM_BOUNDARY


VERSION_INFO = UrbanPiDiTVersionInfo()


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_RUN_NAME",
    "DIAGNOSTIC_PREFIXES",
    "METHOD_FAMILY",
    "PACKAGE_DIRECTORY_NAME",
    "RELEASE_NAME",
    "RELEASE_SLUG",
    "UrbanPiDiTVersionInfo",
    "VERSION",
    "VERSION_INFO",
    "VERSION_NAME",
    "VERSION_SHORT",
]