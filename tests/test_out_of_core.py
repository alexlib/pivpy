import dask.array

from pivpy import io


def test_large_directory_zarr_conversion_stays_lazy(tmp_path):
    """500+ frame dataset written to Zarr, reopened lazily (dask-backed),
    without ever materializing all frames into one eager in-memory Dataset."""
    ds = io.create_sample_Dataset(n_frames=500, rows=8, cols=8)
    zarr_path = tmp_path / "large.zarr"
    io.save_piv(ds, zarr_path, format="zarr", chunks={"t": 1})

    lazy = io.read_directory_lazy(zarr_path)
    assert isinstance(lazy["u"].data, dask.array.Array)
    assert lazy.sizes["t"] == 500

    # A temporal reduction works without an explicit upfront .load()/.compute().
    mean_u = lazy["u"].mean("t").compute()
    assert mean_u.shape == (8, 8)
