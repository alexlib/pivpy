import numpy as np
import pytest

from pivpy import io


pytest.importorskip("scipy", reason="requires SciPy")
pytest.importorskip("scipy.signal", reason="requires scipy.signal")


def test_filterf_pivmat_gauss_same_keeps_shape():
    ds = io.create_sample_Dataset(n_frames=2, rows=11, cols=13, noise_sigma=0.0)
    before = ds["u"].values.copy()

    out = ds.piv.filterf(1.0, "gauss", "same")
    assert out.sizes["y"] == ds.sizes["y"]
    assert out.sizes["x"] == ds.sizes["x"]
    assert out.sizes["t"] == ds.sizes["t"]

    # Should run and generally modify something (not a strict guarantee, but
    # a good smoke-check for convolution path).
    assert not np.allclose(out["u"].values, before)


def test_filterf_pivmat_gauss_valid_reduces_shape_and_coords():
    rows, cols = 15, 20
    ds = io.create_sample_Dataset(n_frames=1, rows=rows, cols=cols, noise_sigma=0.0)

    # For filtsize=1, kernel size is 1 + 2*ceil(3.5*1) = 9
    out = ds.piv.filterf(1.0, "gauss")
    assert out.sizes["y"] == rows - 8
    assert out.sizes["x"] == cols - 8

    # Coords should be truncated consistently.
    assert np.allclose(out["y"].values, ds["y"].values[4:-4])
    assert np.allclose(out["x"].values, ds["x"].values[4:-4])


def test_filterf_pivmat_flat_even_kernel_valid_coords():
    rows, cols = 10, 12
    ds = io.create_sample_Dataset(n_frames=1, rows=rows, cols=cols, noise_sigma=0.0)

    # flat kernel uses size ceil(filtsize); with filtsize=4 => k=4, output N-3
    out = ds.piv.filterf(4.0, "flat")
    assert out.sizes["y"] == rows - 3
    assert out.sizes["x"] == cols - 3

    # Asymmetric truncation for even kernel: left=1, right=2.
    assert np.allclose(out["y"].values, ds["y"].values[1:-2])
    assert np.allclose(out["x"].values, ds["x"].values[1:-2])


def test_filterf_pivmat_nan_aware_constant_field_stays_constant():
    ds = io.create_sample_Dataset(n_frames=1, rows=11, cols=11, noise_sigma=0.0)
    ds["u"][:] = 5.0
    ds["v"][:] = 5.0

    # Single NaN should not bias normalization.
    ds["u"][5, 5, 0] = np.nan

    out = ds.piv.filterf(1.0, "gauss", "same")
    assert np.isfinite(out["u"].values[6, 5, 0])
    assert np.isclose(out["u"].values[6, 5, 0], 5.0, atol=1e-12)


def test_filterf_pivmat_kwargs_rejected_for_float_filtsize():
    ds = io.create_sample_Dataset(n_frames=1, rows=9, cols=9, noise_sigma=0.0)
    with pytest.raises(TypeError):
        ds.piv.filterf(1.0, "gauss", "same", truncate=3.5)
