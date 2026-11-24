"""Golden-master helpers: flatten nested results to named arrays and compare them."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

GOLDEN = Path(__file__).parent / "golden"
type Flat = dict[str, NDArray[np.generic]]


def flatten(obj: object, prefix: str = "") -> Flat:
    """Nested mappings, sequences, arrays and scalars as {path: array}, leaves kept exact."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return flatten(dataclasses.asdict(obj), prefix)
    if isinstance(obj, Mapping):
        out: Flat = {}
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}/{key}"))
        return out
    if isinstance(obj, list | tuple) and not _is_number_row(obj):
        out = {}
        for i, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}/{i}"))
        return out
    return {prefix: np.asarray(obj)}


def _is_number_row(seq: list[object] | tuple[object, ...]) -> bool:
    return all(isinstance(v, int | float | np.generic) and not isinstance(v, bool) for v in seq)


def load(name: str) -> Flat:
    data = np.load(GOLDEN / f"{name}.npz", allow_pickle=False)
    return {key: data[key] for key in data.files}


def subtree(flat: Flat, prefix: str) -> Flat:
    """Entries at or below ``prefix`` with the prefix removed from their keys."""
    return {
        key[len(prefix) :]: value
        for key, value in flat.items()
        if key == prefix or key.startswith(prefix + "/")
    }


def assert_matches(actual: object, expected: Flat, *, rtol: float, atol: float) -> None:
    got = flatten(actual)
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        if value.dtype.kind in "US":
            assert got[key].tolist() == value.tolist(), key
        else:
            np.testing.assert_allclose(
                as_f64(got[key]), as_f64(value), rtol=rtol, atol=atol, err_msg=key
            )


def max_abs_difference(actual: object, expected: Flat) -> float:
    got = flatten(actual)
    return max(
        (
            float(np.max(np.abs(as_f64(got[k]) - as_f64(v))))
            for k, v in expected.items()
            if v.dtype.kind in "fiu" and v.size
        ),
        default=0.0,
    )


def as_f64(a: NDArray[np.generic]) -> NDArray[np.float64]:
    return np.asarray(a, dtype=np.float64)


def as_f32(a: NDArray[np.generic]) -> NDArray[np.float32]:
    return np.asarray(a, dtype=np.float32)


def as_u8(a: NDArray[np.generic]) -> NDArray[np.uint8]:
    return np.asarray(a, dtype=np.uint8)
