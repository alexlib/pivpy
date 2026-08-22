import numpy as np
import pytest
import xarray as xr
import pivpy.pivpy  # registers .piv
from pivpy import io
from pivpy.compute_funcs import reynolds_decomposition
from pivpy.synthetic import vortex_pair
import pivpy.app as piv_app


def test_stream_statistics_eager_dataset():
    """Validates that online streaming statistics match exact in-memory Reynolds decomposition."""
    ds = vortex_pair(n_frames=20, n=(16, 16))
    
    exact = reynolds_decomposition(ds)
    streamed = io.stream_statistics(ds)
    
    np.testing.assert_allclose(streamed["u_mean"].values, exact["u_mean"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed["v_mean"].values, exact["v_mean"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed["uu_prime"].values, exact["uu_prime"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed["vv_prime"].values, exact["vv_prime"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed["uv_prime"].values, exact["uv_prime"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed["tke"].values, exact["tke"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed["intensity_u"].values, exact["intensity_u"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed["intensity_v"].values, exact["intensity_v"].values, rtol=1e-10)


def test_stream_statistics_sequence_and_files(tmp_path):
    """Validates streaming over a sequence of individual frame datasets and files."""
    ds = vortex_pair(n_frames=10, n=(12, 12))
    frames_list = [ds.isel(t=i) for i in range(10)]
    
    exact = reynolds_decomposition(ds)
    streamed_seq = io.stream_statistics(frames_list)
    
    np.testing.assert_allclose(streamed_seq["u_mean"].values, exact["u_mean"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed_seq["tke"].values, exact["tke"].values, rtol=1e-10)
    
    # Save as CSV per frame
    file_list = []
    for i, frame in enumerate(frames_list):
        fp = tmp_path / f"frame_{i:03d}.csv"
        io.save_piv(frame, fp, format="csv")
        file_list.append(fp)
        
    streamed_files = io.stream_statistics(file_list)
    np.testing.assert_allclose(streamed_files["u_mean"].values, exact["u_mean"].values, rtol=1e-5)
    np.testing.assert_allclose(streamed_files["tke"].values, exact["tke"].values, rtol=1e-5)


def test_stream_statistics_lazy_zarr(tmp_path):
    """Validates streaming reduction on Zarr archive."""
    ds = vortex_pair(n_frames=15, n=(10, 10))
    zarr_dir = tmp_path / "stream_test.zarr"
    io.save_piv(ds, zarr_dir, format="zarr", chunks={"t": 1})
    
    streamed_zarr = io.stream_statistics(zarr_dir)
    exact = reynolds_decomposition(ds)
    
    np.testing.assert_allclose(streamed_zarr["u_mean"].values, exact["u_mean"].values, rtol=1e-10)
    np.testing.assert_allclose(streamed_zarr["tke"].values, exact["tke"].values, rtol=1e-10)


def test_accessor_stream_statistics():
    """Validates ds.piv.stream_statistics() accessor method."""
    ds = vortex_pair(n_frames=12, n=(10, 10))
    streamed = ds.piv.stream_statistics()
    exact = reynolds_decomposition(ds)
    
    assert "u_mean" in streamed.data_vars
    assert "tke" in streamed.data_vars
    np.testing.assert_allclose(streamed["tke"].values, exact["tke"].values, rtol=1e-10)


def test_marimo_app_loaded():
    """Validates that pivpy.app Marimo application object is created successfully."""
    assert piv_app.HAS_MARIMO is True
    assert piv_app.app is not None
    assert hasattr(piv_app, "launch_app")
