"""Test stellatscale."""

import stellatscale


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(stellatscale.__name__, str)
