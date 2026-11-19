import numpy as np
import pytest
import xarray as xr

from pivpy import io


def test_tke_budget_basic():
    ds = io.create_sample_Dataset(n_frames=6, rows=12, cols=12)
    out = ds.piv.tke_budget(nu=1e-3)
    for v in ["k", "production", "dissipation", "advection", "turbulent_transport", "viscous_diffusion", "residual"]:
        assert v in out.data_vars
        assert out[v].shape == (12, 12)
    assert np.isfinite(out["k"]).all()
    assert np.isfinite(out["production"]).all()


def test_tke_budget_requires_time():
    ds = io.create_sample_Dataset(n_frames=1, rows=8, cols=8)
    # single frame has no t? create_sample_Dataset default n_frames maybe 2; force by slicing
    ds1 = ds.isel(t=0)
    with pytest.raises(ValueError):
        ds1.piv.tke_budget()


def test_tke_budget_no_nu_nan_dissipation():
    ds = io.create_sample_Dataset(n_frames=4, rows=8, cols=8)
    out = ds.piv.tke_budget(nu=None)
    assert np.isnan(out["dissipation"]).all()
