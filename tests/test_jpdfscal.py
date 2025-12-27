import numpy as np
import pytest
import xarray as xr

import pivpy.pivpy  # registers accessor
from pivpy import graphics
from pivpy.compute_funcs import jpdfscal


def test_jpdfscal_basic_counts_nbin3():
    # Values map exactly to bins [-1, 0, 1]
    a = xr.DataArray(np.array([[-1.0, 0.0], [1.0, 0.0]]), dims=("y", "x"))
    b = xr.DataArray(np.array([[0.0, -1.0], [1.0, 0.0]]), dims=("y", "x"))

    out = jpdfscal(a, b, nbin=3)
    assert out["hi"].shape == (3, 3)

    # Expected pairs:
    # (-1, 0), (0, -1), (1, 1), (0, 0)
    # bin indices: -1->0, 0->1, 1->2
    expected = np.zeros((3, 3), dtype=float)
    expected[0, 1] += 1
    expected[1, 0] += 1
    expected[2, 2] += 1
    expected[1, 1] += 1

    assert np.array_equal(out["hi"].values, expected)


def test_jpdfscal_ignores_nonfinite_pairs():
    a = xr.DataArray(np.array([[np.nan, 0.0], [1.0, 2.0]]), dims=("y", "x"))
    b = xr.DataArray(np.array([[0.0, np.nan], [1.0, 2.0]]), dims=("y", "x"))

    out = jpdfscal(a, b, nbin=5)
    assert float(out["hi"].sum()) == 2.0


def test_jpdfscal_requires_odd_nbin():
    a = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
    with pytest.raises(ValueError):
        jpdfscal(a, a, nbin=4)


def test_jpdfscal_disp_returns_fig_ax():
    a = xr.DataArray(np.array([[-1.0, 0.0], [1.0, 0.0]]), dims=("y", "x"))
    b = xr.DataArray(np.array([[0.0, -1.0], [1.0, 0.0]]), dims=("y", "x"))
    out = jpdfscal(a, b, nbin=3)

    fig, ax = graphics.jpdfscal_disp(out)
    assert fig is not None
    assert ax is not None
    fig.canvas.draw()
    import matplotlib.pyplot as plt

    plt.close(fig)
