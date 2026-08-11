"""Attrs-preservation checks: does delta_t/units/files survive accessor calls?"""
import pytest

from pivpy import io
import pivpy.pivpy  # noqa: F401 -- registers the .piv accessor


def _base_dataset():
    ds = io.create_sample_Dataset(n_frames=5, rows=10, cols=10)
    ds.attrs["delta_t"] = 0.5
    return ds


def _assert_core_attrs_preserved(ds):
    assert ds.attrs.get("delta_t") == 0.5
    assert "files" in ds.attrs
    assert ds["x"].attrs.get("units")
    assert ds["u"].attrs.get("units")


@pytest.mark.parametrize(
    "apply",
    [
        lambda ds: ds.piv.crop([0, 5, 0, 5]),
        lambda ds: ds.piv.pan(1.0, -1.0),
        lambda ds: ds.piv.filterf([0.5, 0.5, 0.0]),
        lambda ds: ds.piv.flipf("x"),
        lambda ds: ds.piv.vorticity(name="vort"),
        lambda ds: ds.piv.strain(name="strain"),
        lambda ds: ds.piv.averf(),
        lambda ds: ds.piv.addnoisef(eps=0.1, seed=0),
    ],
    ids=["crop", "pan", "filterf", "flipf", "vorticity", "strain", "averf", "addnoisef"],
)
def test_accessor_method_preserves_core_attrs(apply):
    ds = _base_dataset()
    out = apply(ds)
    _assert_core_attrs_preserved(out)
