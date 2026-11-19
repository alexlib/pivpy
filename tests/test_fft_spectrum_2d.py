import numpy as np

from pivpy import io


def test_fft_spectrum_2d_basic():
    ds = io.create_sample_Dataset(n_frames=2, rows=16, cols=16)
    out = ds.piv.fft_spectrum(component="u")
    assert "E_radial" in out.data_vars
    assert out["E_radial"].shape[0] == out["k"].shape[0]
    assert np.all(np.isfinite(out["E_radial"].values))
