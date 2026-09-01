"""Smoke test for the dqnpf-intraday-base image import surface.

Property DQNPF-IMG-1: All required modules importable.

The actual built image is verified via a ``RUN python -c "..."`` step in the
Dockerfile (see ``Dockerfile.dqnpf-intraday-base``). These tests verify two
things from the host:

1. The import surface itself works in the host environment, so the same
   imports will succeed once the parent packages are pip-installed in the
   image.
2. The Dockerfile actually contains the dual-import smoke test step.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


_REQUIRED_MODULES = [
    # dqnpf deliberately does not re-export at the package root (see its
    # __init__), so check the module that actually defines the symbol.
    ("dqnpf.integration", "IntegrationLayer"),
    ("deepqnetwork.advisor", "DQNAdvisor"),
    ("probabilisticforecaster.inference", "ForecasterInference"),
]


@pytest.mark.parametrize("module_name,attr", _REQUIRED_MODULES)
def test_required_module_importable(module_name: str, attr: str) -> None:
    """Every module the predictor image relies on must import in-process."""
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), f"{module_name} missing {attr}"


def test_dual_import_smoke_runs_in_process() -> None:
    """The exact import line baked into the Dockerfile must succeed locally."""
    from dqnpf.integration import IntegrationLayer  # noqa: F401
    from deepqnetwork.advisor import DQNAdvisor  # noqa: F401
    from probabilisticforecaster.inference import ForecasterInference  # noqa: F401


def test_dockerfile_contains_import_smoke_step() -> None:
    """Dockerfile must contain the RUN step that verifies imports at build."""
    dockerfile = (
        Path(__file__).resolve().parents[1] / "Dockerfile.dqnpf-intraday-base"
    )
    assert dockerfile.exists(), f"missing {dockerfile}"
    content = dockerfile.read_text()
    assert "from dqnpf.integration import IntegrationLayer" in content
    assert "from deepqnetwork.advisor import DQNAdvisor" in content
    assert "from probabilisticforecaster.inference import ForecasterInference" in content


def test_dockerfile_extends_dqn_base() -> None:
    """The image must build on top of dqn-base (Req 20.1, 20.4)."""
    dockerfile = (
        Path(__file__).resolve().parents[1] / "Dockerfile.dqnpf-intraday-base"
    )
    content = dockerfile.read_text()
    assert "ARG DQN_BASE_IMAGE" in content
    assert "FROM ${DQN_BASE_IMAGE}" in content
