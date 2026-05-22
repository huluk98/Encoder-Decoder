from __future__ import annotations

from encoder_decoder.evaluate_exact import normalize_for_exact


def test_normalize_for_exact_defaults_to_strip_only() -> None:
    assert normalize_for_exact("  Answer  ") == "Answer"
    assert normalize_for_exact("A   B") == "A   B"


def test_normalize_for_exact_can_collapse_and_lowercase() -> None:
    assert (
        normalize_for_exact("  A   B  ", lowercase=True, collapse_whitespace=True)
        == "a b"
    )

