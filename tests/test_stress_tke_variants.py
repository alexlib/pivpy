import numpy as np

from pivpy import io


def test_reynolds_stress_std():
    ds = io.create_sample_Dataset(n_frames=4, rows=8, cols=8)
    out = ds.piv.reynolds_stress_std()
    assert "uu_std" in out.data_vars and "tke_std" in out.data_vars


def test_tke_std():
    ds = io.create_sample_Dataset(n_frames=4, rows=8, cols=8)
    da = ds.piv.tke_std()
    assert da.shape == (8, 8)


def test_stress_line_profile():
    ds = io.create_sample_Dataset(n_frames=4, rows=8, cols=8)
    lp = ds.piv.stress_line_profile(0, 0, 1, 1, component="uu")
    assert "value" in lp.data_vars
    assert lp.sizes["s"] == 50
