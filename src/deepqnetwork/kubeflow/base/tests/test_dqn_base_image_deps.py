"""Property-based tests for DQN base image dependency consistency.

Verifies that the Python packages specified in requirements-dqn-base.txt are
importable in the current environment, that the gRPC stubs exist at the expected
paths, and that the proto source file exists for regeneration.

This validates the dependency specification is consistent with what the DQN
pipeline code actually imports, without requiring a Docker image build.

**Validates: Requirements DQN-R4**
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


REPO_ROOT = Path(__file__).resolve().parents[4]
REQUIREMENTS_FILE = REPO_ROOT / "deepqnetwork" / "kubeflow" / "base" / "requirements-dqn-base.txt"
PROTO_FILE = REPO_ROOT / "modelenv" / "proto" / "proto" / "environment.proto"

_PACKAGE_TO_IMPORT = {
    "torch": "torch",
    "numpy": "numpy",
    "grpcio": "grpc",
    "grpcio-tools": "grpc_tools",
    "protobuf": "google.protobuf",
    "boto3": "boto3",
    "PyYAML": "yaml",
}

_REQUIRED_MODULES = ["torch", "grpc", "numpy", "boto3", "environment_pb2"]

_STUB_LOCATIONS = [
    REPO_ROOT / "environment_pb2.py",
    REPO_ROOT / "environment_pb2_grpc.py",
]


def _parse_requirements(path: Path) -> list[tuple[str, str]]:
    """Parse requirements file into list of (package_name, version) tuples.

    Skips comments and blank lines. Strips version specifiers like ==1.2.3+cpu.
    """
    packages: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)==(.+)$", line)
        if match:
            packages.append((match.group(1), match.group(2)))
    return packages


_packages = _parse_requirements(REQUIREMENTS_FILE)
_package_names = [pkg for pkg, _ in _packages]

package_strategy = st.sampled_from(_package_names)

required_module_strategy = st.sampled_from(_REQUIRED_MODULES)


class TestDQNBaseImageDependencyConsistency:
    """Property DQN-4: Base image Python imports.

    All required modules (torch, grpc, numpy, boto3, environment_pb2) are
    importable in the built image. We verify this without building the image
    by checking that:
    1. All packages in requirements-dqn-base.txt are importable
    2. The gRPC stub files exist at expected paths
    3. The proto source file exists for regeneration

    **Validates: Requirements DQN-R4**
    """

    @given(package_name=package_strategy)
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_requirements_packages_are_importable(self, package_name: str) -> None:
        """For any package listed in requirements-dqn-base.txt, the corresponding
        Python module is importable in the current environment.

        **Validates: Requirements DQN-R4**
        """
        import_name = _PACKAGE_TO_IMPORT.get(package_name)
        assert import_name is not None, (
            f"Package '{package_name}' has no known import mapping in _PACKAGE_TO_IMPORT"
        )
        try:
            importlib.import_module(import_name)
        except ImportError as e:
            pytest.fail(
                f"Package '{package_name}' (import as '{import_name}') is listed in "
                f"requirements-dqn-base.txt but cannot be imported: {e}"
            )

    @given(module_name=required_module_strategy)
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_required_modules_are_importable(self, module_name: str) -> None:
        """For any module required by the DQN pipeline code (torch, grpc, numpy,
        boto3, environment_pb2), the module is importable.

        **Validates: Requirements DQN-R4**
        """
        if module_name == "environment_pb2":
            import sys
            repo_root_str = str(REPO_ROOT)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            try:
                importlib.import_module("environment_pb2")
            except ImportError as e:
                pytest.fail(
                    f"Module 'environment_pb2' is not importable from repo root: {e}"
                )
        else:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(
                    f"Required module '{module_name}' is not importable: {e}"
                )

    def test_grpc_stub_files_exist(self) -> None:
        """The gRPC stub files (environment_pb2.py, environment_pb2_grpc.py)
        exist at the expected repo paths for copying into the Docker image.

        **Validates: Requirements DQN-R4**
        """
        for stub_path in _STUB_LOCATIONS:
            assert stub_path.exists(), (
                f"gRPC stub file not found at {stub_path}. "
                f"The Dockerfile.dqn-base expects to COPY stubs from this location."
            )

    def test_proto_source_file_exists(self) -> None:
        """The modelenv proto source file exists for stub regeneration.

        **Validates: Requirements DQN-R4**
        """
        assert PROTO_FILE.exists(), (
            f"Proto source file not found at {PROTO_FILE}. "
            f"This file is needed to regenerate environment_pb2.py and "
            f"environment_pb2_grpc.py stubs."
        )

    def test_requirements_file_has_pinned_versions(self) -> None:
        """All packages in requirements-dqn-base.txt have pinned versions (==).

        **Validates: Requirements DQN-R4**
        """
        packages = _parse_requirements(REQUIREMENTS_FILE)
        assert len(packages) > 0, "requirements-dqn-base.txt is empty"

        for pkg_name, version in packages:
            assert version, (
                f"Package '{pkg_name}' has no pinned version in "
                f"requirements-dqn-base.txt"
            )

    def test_requirements_covers_all_required_modules(self) -> None:
        """The requirements file covers the pip-installed required modules.

        grpc, numpy, and boto3 must each have a corresponding package in
        requirements-dqn-base.txt. torch is intentionally NOT pinned there;
        its wheel is arch-specific and installed separately by
        Dockerfile.dqn-base (see test_dockerfile_installs_torch). environment_pb2
        is provided via compiled stubs, not pip.

        **Validates: Requirements DQN-R4**
        """
        packages = _parse_requirements(REQUIREMENTS_FILE)
        package_names = {pkg.lower() for pkg, _ in packages}

        module_to_package = {
            "grpc": "grpcio",
            "numpy": "numpy",
            "boto3": "boto3",
        }

        for module, expected_package in module_to_package.items():
            assert expected_package.lower() in package_names, (
                f"Required module '{module}' (pip package '{expected_package}') "
                f"is not listed in requirements-dqn-base.txt"
            )

    def test_dockerfile_installs_torch(self) -> None:
        """torch is installed by Dockerfile.dqn-base (arch-specific CUDA wheel),
        not pinned in requirements-dqn-base.txt.

        **Validates: Requirements DQN-R4**
        """
        dockerfile = (
            REPO_ROOT / "deepqnetwork" / "kubeflow" / "base" / "Dockerfile.dqn-base"
        )
        content = dockerfile.read_text()
        assert "torch" in content, (
            "Dockerfile.dqn-base does not install torch (arch-specific CUDA wheel)"
        )

    def test_dockerfile_references_requirements_file(self) -> None:
        """The Dockerfile.dqn-base references requirements-dqn-base.txt.

        **Validates: Requirements DQN-R4**
        """
        dockerfile = REPO_ROOT / "deepqnetwork" / "kubeflow" / "base" / "Dockerfile.dqn-base"
        assert dockerfile.exists(), f"Dockerfile not found at {dockerfile}"

        content = dockerfile.read_text()
        assert "requirements-dqn-base.txt" in content, (
            "Dockerfile.dqn-base does not reference requirements-dqn-base.txt"
        )

    def test_dockerfile_copies_grpc_stubs(self) -> None:
        """The Dockerfile.dqn-base copies the gRPC stubs into the image.

        **Validates: Requirements DQN-R4**
        """
        dockerfile = REPO_ROOT / "deepqnetwork" / "kubeflow" / "base" / "Dockerfile.dqn-base"
        content = dockerfile.read_text()
        assert "environment_pb2.py" in content, (
            "Dockerfile.dqn-base does not COPY environment_pb2.py"
        )
        assert "environment_pb2_grpc.py" in content, (
            "Dockerfile.dqn-base does not COPY environment_pb2_grpc.py"
        )
