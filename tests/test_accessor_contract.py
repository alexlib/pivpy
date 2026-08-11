"""Checks for the Phase 3 accessor cleanups: no silent in-place mutation on
__mul__/__div__, a visible warning on the 'w' name-collision footgun, and a
DeprecationWarning on the still-mutating crop()/vec2scal() methods."""
import pytest

from pivpy import io
import pivpy.pivpy  # noqa: F401 -- registers the .piv accessor


def test_mul_does_not_mutate_original_dataset():
    ds = io.create_sample_Dataset(n_frames=2)
    original_u = ds["u"].values.copy()

    scaled = ds.piv * 2.0

    assert (ds["u"].values == original_u).all()
    assert (scaled["u"].values == original_u * 2.0).all()


def test_div_does_not_mutate_original_dataset():
    ds = io.create_sample_Dataset(n_frames=2)
    original_u = ds["u"].values.copy()

    scaled = ds.piv / 2.0

    assert (ds["u"].values == original_u).all()
    assert (scaled["u"].values == original_u / 2.0).all()


def test_scalar_overwrite_warns_by_default():
    ds = io.create_sample_Dataset(n_frames=2)
    ds = ds.piv.vorticity()  # writes "w"
    with pytest.warns(UserWarning, match="already exists"):
        ds.piv.strain()  # also defaults to "w" -- should warn


def test_scalar_no_warning_with_distinct_names():
    ds = io.create_sample_Dataset(n_frames=2)
    ds = ds.piv.vorticity(name="vort")
    with warnings_none():
        ds.piv.strain(name="strain")


def test_crop_warns_deprecation():
    ds = io.create_sample_Dataset(n_frames=2, rows=10, cols=10)
    with pytest.warns(DeprecationWarning, match="crop"):
        ds.piv.crop([0, 5, 0, 5])


def test_vec2scal_warns_deprecation():
    ds = io.create_sample_Dataset(n_frames=2)
    with pytest.warns(DeprecationWarning, match="vec2scal"):
        ds.piv.vec2scal("vorticity")


class warnings_none:
    """Context manager asserting no warnings are raised inside the block."""

    def __enter__(self):
        import warnings as _w

        self._catcher = _w.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        _w.simplefilter("always")
        return self

    def __exit__(self, *exc_info):
        self._catcher.__exit__(*exc_info)
        assert not self._records, [str(r.message) for r in self._records]
