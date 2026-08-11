import dask.array
import xarray as xr

from pivpy import io


def test_zarr_roundtrip(tmp_path):
    ds = io.create_sample_Dataset()
    path = tmp_path / "sample.zarr"

    io.save_piv(ds, path, format="zarr")
    ds2 = io.read_piv(path)

    xr.testing.assert_identical(ds, ds2)


def test_zarr_lazy_loading(tmp_path):
    ds = io.create_sample_Dataset(n_frames=10)
    path = tmp_path / "sample_lazy.zarr"

    io.save_piv(ds, path, format="zarr", chunks={"t": 1})
    ds2 = io.read_piv(path, chunks="auto")

    assert isinstance(ds2["u"].data, dask.array.Array)
