import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import MockSource  # noqa: E402
from data.mock_generator import generate_all  # noqa: E402

SMALL = {"train": 240, "cal": 240, "test": 240, "ood": 120}


@pytest.fixture(scope="session")
def mock_root(tmp_path_factory) -> Path:
    """A small mock dataset, generated once per test session."""
    root = tmp_path_factory.mktemp("mock")
    generate_all(root, sizes=SMALL, seed=0)
    return root


@pytest.fixture(scope="session")
def sources(mock_root) -> dict:
    return {s: MockSource(mock_root, s) for s in SMALL}
