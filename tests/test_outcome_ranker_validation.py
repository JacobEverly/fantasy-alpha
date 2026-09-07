import pytest

from evals.outcome_ranker_validation import sha256


def test_sha256_is_content_addressed(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"one")
    first = sha256(path)
    path.write_bytes(b"two")
    assert sha256(path) != first


def test_validation_module_exposes_no_fit_function():
    import evals.outcome_ranker_validation as module

    assert not hasattr(module, "fit")
    with pytest.raises(AttributeError):
        getattr(module, "train")
