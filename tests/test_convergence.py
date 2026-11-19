import numpy as np
import pytest

from pivpy import io


def test_convergence_basic():
    ds = io.create_sample_Dataset(n_frames=8, rows=8, cols=8)
    out = ds.piv.convergence()
    assert "n" in out.coords
    assert out.sizes["n"] == 8
    for q in ["u", "v", "uu", "vv", "uv", "k"]:
        assert q in out.data_vars
        assert out[q].attrs.get("convergence_n") is not None


def test_convergence_requires_time():
    ds = io.create_sample_Dataset(n_frames=2, rows=6, cols=6).isel(t=0)
    with pytest.raises(ValueError):
        ds.piv.convergence()
