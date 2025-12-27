import numpy as np
import pytest

from pivpy import io


def test_im2pivmat_basic_shape_coords_dtype_units():
    im = (np.arange(12).reshape(3, 4) + 0.5).astype(np.float64)
    ds = io.im2pivmat(im)

    assert "w" in ds
    assert ds["w"].dims == ("y", "x", "t")
    assert ds["w"].shape == (3, 4, 1)

    assert np.allclose(ds["w"].isel(t=0).values, im.astype(np.float32))
    assert ds["w"].dtype == np.float32

    assert np.allclose(ds["x"].values, [1, 2, 3, 4])
    assert np.allclose(ds["y"].values, [1, 2, 3])
    assert np.allclose(ds["t"].values, [0.0])

    assert ds["x"].attrs.get("units") == "au"
    assert ds["y"].attrs.get("units") == "au"
    assert ds["w"].attrs.get("units") == "au"
    assert ds.attrs.get("source") == "image"


def test_im2pivmat_custom_coords_and_metadata():
    im = np.zeros((2, 3), dtype=np.uint8)
    x = np.array([10.0, 20.0, 30.0])
    y = np.array([-1.0, -2.0])
    ds = io.im2pivmat(im, x=x, y=y, namew="I", unit="pix")

    assert np.allclose(ds["x"].values, x)
    assert np.allclose(ds["y"].values, y)
    assert ds["w"].attrs.get("long_name") == "I"
    assert ds["w"].attrs.get("units") == "pix"


def test_im2pivmat_rejects_non_2d_input():
    with pytest.raises(ValueError):
        io.im2pivmat(np.zeros((2, 2, 2)))


def test_im2pivmat_rejects_bad_coord_lengths():
    im = np.zeros((2, 3))
    with pytest.raises(ValueError):
        io.im2pivmat(im, x=[1, 2])
    with pytest.raises(ValueError):
        io.im2pivmat(im, y=[1, 2, 3])
