"""Tests for Γ1/Γ2 tutorial code.

This file originally started life as a notebook-style tutorial script.

Pytest will import any `test_*.py` module during collection. To keep tests fast and
reliable, we must NOT execute the tutorial (plots, file IO, OpenPIV processing)
at import time.

Instead:
- keep the demo runnable via `python -m tests.test_gamma` (or direct execution)
- provide a small, fast smoke test that validates Γ1/Γ2 run without warnings
"""

from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np

# Make this module safe in headless environments (CI).
import matplotlib

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pivpy import io, pivpy  # noqa: F401  (registers the .piv accessor)


def _safe_levels(da, n: int = 1000):
    vmin = float(da.min().values)
    vmax = float(da.max().values)
    if vmin == vmax:
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-12
        return [vmin - eps, vmax + eps]
    return np.linspace(vmin, vmax, n)


def _fields_dir() -> pathlib.Path:
    from importlib import resources

    return pathlib.Path(
        resources.files("pivpy")
        / "data"
        / "openpiv_txt"
        / "Gamma1_Gamma2_tutorial_notebook"
    )


def run_demo(*, make_plots: bool = True) -> None:
    """Run the original tutorial-style demo.

    This is intentionally *not* executed during pytest collection.
    """

    fields_dir = _fields_dir()
    n = 4

    data = io.load_directory(
        path=fields_dir,
        basename="OpenPIVtxtFilePair?",
        ext=".txt",
    )

    data.piv.Γ1(n, convCoords=True)
    data.piv.Γ2(n, convCoords=False)

    if not make_plots:
        return

    # For t=0
    levels_g1 = _safe_levels(data["Γ1"].isel(t=0))
    levels_g2 = _safe_levels(data["Γ2"].isel(t=0))
    fig_g, ax_g = plt.subplots(nrows=1, ncols=2, clear=True, figsize=(15, 10))
    data["Γ1"].isel(t=0).plot.contourf(x="x", y="y", levels=levels_g1, cmap=plt.get_cmap("RdYlBu"), ax=ax_g[0])
    data["Γ2"].isel(t=0).plot.contourf(x="x", y="y", levels=levels_g2, cmap=plt.get_cmap("RdYlBu"), ax=ax_g[1])


def test_gamma_smoke_no_runtimewarning() -> None:
    """Basic, fast smoke test for Γ1/Γ2.

    - Must be quick (no OpenPIV image processing)
    - Must not write files
    - Should not emit RuntimeWarning from empty neighborhoods
    """

    data = io.create_sample_Dataset(n_frames=2, rows=6, cols=6)

    # Force an all-NaN case to exercise the "empty neighborhood" path.
    data["u"][:] = np.nan
    data["v"][:] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = data.piv.Γ1(n=1)
        out = out.piv.Γ2(n=1)

    assert "Γ1" in out
    assert "Γ2" in out
    assert np.all(np.isfinite(out["Γ1"].to_numpy()))
    assert np.all(np.isfinite(out["Γ2"].to_numpy()))


if __name__ == "__main__":
    run_demo(make_plots=True)


