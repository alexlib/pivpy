"""Tests for the pyTurb-derived temporal helpers on xarray PIV datasets.

Original algorithms: pyTurb's flowstats.py by Alec Petersen
(https://github.com/ajpetersen25/pyTurb).
"""
import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid

from pivpy.schema import build_dataset
import pivpy.pivpy  # noqa: F401  (registers the .piv accessor)
from pivpy.compute_funcs import moving_average, probe_esd


def _tone_dataset(n_t=4096, fs=100.0, f0=10.0, noise=0.01, seed=0):
    """Uniform field whose u-component is a sinusoid at f0 (sampled at fs)."""
    t = np.arange(n_t) / fs
    rng = np.random.default_rng(seed)
    sig = np.sin(2 * np.pi * f0 * t) + noise * rng.standard_normal(n_t)
    u = np.broadcast_to(sig, (4, 4, n_t)).copy()
    v = np.zeros_like(u)
    x = np.arange(4.0)
    y = np.arange(4.0)
    return build_dataset(x, y, u, v, t=t)


def _noise_dataset(n_t=2048, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((4, 4, n_t))
    return build_dataset(np.arange(4.0), np.arange(4.0), u, np.zeros_like(u), t=np.arange(n_t, dtype=float))


def test_probe_esd_peak_at_tone_frequency():
    ds = _tone_dataset(fs=100.0, f0=10.0)
    out = probe_esd(ds, 1.5, 1.5, variable="u", windows=8)

    assert isinstance(out, xr.Dataset)
    assert "esd" in out and out["esd"].dims == ("f",)
    f = out["f"].values
    E = out["esd"].values
    assert np.all(E >= 0)
    df = f[1] - f[0]
    assert abs(f[np.argmax(E)] - 10.0) <= 2 * df
    assert out["esd"].attrs["fs"] == pytest.approx(100.0)


def test_probe_esd_integral_matches_variance():
    ds = _noise_dataset()
    out = probe_esd(ds, 1.0, 1.0, variable="u", windows=4)
    integral = trapezoid(out["esd"].values, out["f"].values)
    series = ds["u"].interp(x=1.0, y=1.0).values
    assert integral == pytest.approx(np.var(series), rel=0.1)


def test_probe_esd_applies_low_frequency_cutoff():
    ds = _noise_dataset(n_t=1024)
    windows, fs = 4, 1.0
    out = probe_esd(ds, 1.0, 1.0, windows=windows, fs=fs)
    window_size = int(np.floor(1024 / (windows * 0.5 + 0.5)))
    assert out["f"].values.min() >= fs / window_size - 1e-12


def test_probe_esd_rejects_unknown_variable():
    ds = _noise_dataset()
    with pytest.raises(KeyError):
        probe_esd(ds, 1.0, 1.0, variable="w")


def test_accessor_probe_esd_matches_function():
    ds = _tone_dataset()
    via_accessor = ds.piv.probe_esd(1.5, 1.5, variable="u", windows=8)
    direct = probe_esd(ds, 1.5, 1.5, variable="u", windows=8)
    xr.testing.assert_allclose(via_accessor, direct)


def test_moving_average_valid_window_count():
    ds = _noise_dataset(n_t=10)
    out = moving_average(ds, window=3)
    assert out.sizes["t"] == 8
    expected = ds["u"].isel(t=slice(0, 3)).mean("t")
    np.testing.assert_allclose(out["u"].isel(t=0).values, expected.values)


def test_accessor_moving_average_matches_function():
    ds = _noise_dataset(n_t=12)
    xr.testing.assert_allclose(ds.piv.moving_average(4), moving_average(ds, window=4))


def test_moving_average_rejects_bad_window():
    ds = _noise_dataset(n_t=6)
    with pytest.raises(ValueError):
        moving_average(ds, window=0)
    with pytest.raises(ValueError):
        moving_average(ds, dim="z")
