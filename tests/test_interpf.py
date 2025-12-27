import numpy as np
import pytest

import pivpy.pivpy  # registers accessor
from pivpy import io


def _make_missing_field():
    ds = io.create_sample_Dataset(n_frames=2, rows=8, cols=9)
    out = ds.copy(deep=True)

    # Inject zeros and NaNs (missing data in PIVMAT interpf)
    out["u"] = out["u"].where(~(((out["u"] > 3) & (out["u"] < 6))), 0.0)
    out["v"] = out["v"].where(~(((out["v"] > 2) & (out["v"] < 5))), np.nan)
    return out


@pytest.mark.parametrize("method", [0, 1, 2])
def test_interpf_fills_missing_u_v(method):
    ds = _make_missing_field()

    filled = ds.piv.interpf(method=method, variables=["u", "v"], missing="0nan")

    assert filled["u"].shape == ds["u"].shape
    assert filled["v"].shape == ds["v"].shape

    # u: no zeros should remain where we injected them
    assert np.count_nonzero(filled["u"].values == 0.0) < np.count_nonzero(ds["u"].values == 0.0)

    # v: NaNs should be reduced
    assert np.count_nonzero(~np.isfinite(filled["v"].values)) < np.count_nonzero(~np.isfinite(ds["v"].values))


def test_interpf_missing_selector_nan_only():
    ds = _make_missing_field()
    # Only NaNs are considered missing; zeros in u remain.
    filled = ds.piv.interpf(method=1, variables=["u", "v"], missing="nan")

    assert np.count_nonzero(filled["u"].values == 0.0) == np.count_nonzero(ds["u"].values == 0.0)
    assert np.count_nonzero(~np.isfinite(filled["v"].values)) < np.count_nonzero(~np.isfinite(ds["v"].values))


def test_interpf_raises_on_unknown_variable():
    ds = io.create_sample_Dataset(n_frames=1)
    with pytest.raises(KeyError):
        ds.piv.interpf(variables=["does_not_exist"])
