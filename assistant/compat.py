"""Small compatibility helpers for third-party package API changes."""
from packaging.version import Version


def transformers_dtype_kwargs(transformers_version: str, dtype):
    """Return the correct from_pretrained dtype keyword for Transformers version.

    The `dtype` parameter replaced deprecated `torch_dtype` in Transformers 4.56.
    Keep the legacy branch only for environments below that API threshold.
    """
    key = "dtype" if Version(transformers_version) >= Version("4.56.0") else "torch_dtype"
    return {key: dtype}
