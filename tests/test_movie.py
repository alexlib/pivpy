import numpy as np
import pytest
import xarray as xr

import pivpy.pivpy  # registers the .piv accessor
from pivpy import io
from pivpy import graphics


def test_to_movie_vector_returns_frames_length_and_shape():
    ds = io.create_sample_Dataset(n_frames=3, rows=10, cols=11)

    frames = ds.piv.to_movie(
        None,
        return_frames=True,
        show="vector",
        close=True,
        nthArr=2,
        scalingFactor=10.0,
    )

    assert isinstance(frames, list)
    assert len(frames) == 3
    h, w, c = frames[0].shape
    assert c == 4
    assert h > 0 and w > 0
    assert frames[0].dtype == np.uint8


def test_to_movie_scalar_returns_frames_and_changes_with_time():
    rng = np.random.default_rng(0)
    im0 = rng.random((20, 30))
    im1 = rng.random((20, 30))
    ds0 = io.im2pivmat(im0)
    ds1 = io.im2pivmat(im1)

    # Concatenate into a 2-frame scalar dataset (keep variable name 'w')
    ds = xr.concat([ds0, ds1], dim="t")

    frames = graphics.to_movie(ds, None, return_frames=True, show="scalar", scalar="w", close=True)
    assert len(frames) == 2

    # Frames should differ for different scalar inputs (very likely, but keep robust)
    assert np.mean(np.abs(frames[0].astype(float) - frames[1].astype(float))) > 0.1


def test_imvectomovie_streams_files_and_returns_frames(tmp_path):
    ds = io.create_sample_Dataset(n_frames=3, rows=12, cols=13)

    # Write each frame to a netcdf file; read_piv() supports NetCDF via NetCDFReader.
    for i in range(ds.sizes["t"]):
        fp = tmp_path / f"frame_{i:03d}.nc"
        ds.isel(t=i).to_netcdf(fp)

    frames = graphics.imvectomovie(
        str(tmp_path / "frame_*.nc"),
        None,
        return_frames=True,
        show="vector",
        close=True,
        nthArr=3,
    )

    assert len(frames) == 3


def test_imvectomovie_raises_on_no_match(tmp_path):
    with pytest.raises(FileNotFoundError):
        graphics.imvectomovie(str(tmp_path / "nope_*.nc"), None, return_frames=True)
