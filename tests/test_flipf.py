import numpy as np

from pivpy import io


def test_flipf_x_mirrors_x_and_negates_u_only():
    ds = io.create_sample_Dataset(n_frames=2, rows=4, cols=5, noise_sigma=0.0)
    u0 = ds["u"].values.copy()
    v0 = ds["v"].values.copy()
    x0 = ds["x"].values.copy()
    y0 = ds["y"].values.copy()

    out = ds.piv.flipf("x")

    assert np.allclose(out["x"].values, x0)
    assert np.allclose(out["y"].values, y0)

    # u: mirrored along x and sign-changed
    assert np.allclose(out["u"].values, -u0[:, ::-1, :])
    # v: mirrored along x only
    assert np.allclose(out["v"].values, v0[:, ::-1, :])


def test_flipf_y_mirrors_y_and_negates_v_only():
    ds = io.create_sample_Dataset(n_frames=1, rows=6, cols=3, noise_sigma=0.0)
    u0 = ds["u"].values.copy()
    v0 = ds["v"].values.copy()
    x0 = ds["x"].values.copy()
    y0 = ds["y"].values.copy()

    out = ds.piv.flipf("y")

    assert np.allclose(out["x"].values, x0)
    assert np.allclose(out["y"].values, y0)

    # u: mirrored along y only
    assert np.allclose(out["u"].values, u0[::-1, :, :])
    # v: mirrored along y and sign-changed
    assert np.allclose(out["v"].values, -v0[::-1, :, :])


def test_flipf_xy_mirrors_both_and_negates_u_and_v():
    ds = io.create_sample_Dataset(n_frames=1, rows=4, cols=5, noise_sigma=0.0)
    u0 = ds["u"].values.copy()
    v0 = ds["v"].values.copy()

    out = ds.piv.flipf("xy")

    assert np.allclose(out["u"].values, -u0[::-1, ::-1, :])
    assert np.allclose(out["v"].values, -v0[::-1, ::-1, :])


def test_flipf_empty_string_noop():
    ds = io.create_sample_Dataset(n_frames=1, rows=3, cols=3, noise_sigma=0.0)
    out = ds.piv.flipf("")
    assert np.allclose(out["u"].values, ds["u"].values)
    assert np.allclose(out["v"].values, ds["v"].values)


def test_flipf_invalid_dir_raises():
    ds = io.create_sample_Dataset(n_frames=1, rows=3, cols=3, noise_sigma=0.0)
    try:
        ds.piv.flipf("bad")
    except ValueError:
        return
    assert False, "Expected ValueError"
