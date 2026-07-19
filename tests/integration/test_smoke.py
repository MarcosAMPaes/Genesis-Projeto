from __future__ import annotations

import pytest

from petra import __version__


@pytest.mark.integration
@pytest.mark.smoke
def test_package_smoke() -> None:
    assert __version__ == "0.1.0"
