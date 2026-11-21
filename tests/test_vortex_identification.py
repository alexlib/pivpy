import numpy as np

from pivpy import io


def test_detect_vortices_returns_list():
    ds = io.create_sample_Dataset(n_frames=3, rows=12, cols=12)
    vs = ds.piv.detect_vortices(criterion="gamma2", threshold=0.1)
    assert isinstance(vs, list)


def test_vortex_probability_shape():
    ds = io.create_sample_Dataset(n_frames=3, rows=12, cols=12)
    vp = ds.piv.vortex_probability(threshold=0.1)
    assert vp.shape == (12, 12)
    assert np.all((vp.values >= 0) & (vp.values <= 1))


def test_vortex_probability_no_time_dim():
    ds = io.create_sample_Dataset(n_frames=2, rows=12, cols=12).isel(t=0)
    assert "t" not in ds.dims
    vp = ds.piv.vortex_probability()
    assert vp.shape == (12, 12)
    assert np.all((vp.values == 0) | (vp.values == 1))


def test_vortex_probability_single_frame_with_t():
    ds = io.create_sample_Dataset(n_frames=1, rows=12, cols=12)
    assert ds.sizes["t"] == 1
    vp = ds.piv.vortex_probability()
    assert vp.shape == (12, 12)
    assert np.all((vp.values >= 0) & (vp.values <= 1))


def test_vortex_probability_small_grids():
    for n in (2, 3, 5, 7):
        ds = io.create_sample_Dataset(n_frames=2, rows=n, cols=n)
        vp = ds.piv.vortex_probability()
        assert vp.shape == (n, n)
        assert np.all((vp.values >= 0) & (vp.values <= 1))


def test_vortex_probability_sign_filtering():
    ds = io.create_sample_Dataset(n_frames=3, rows=12, cols=12)
    for sign in ("positive", "negative"):
        vp = ds.piv.vortex_probability(threshold=0.1, sign=sign)
        assert vp.shape == (12, 12)
        assert np.all((vp.values >= 0) & (vp.values <= 1))


def test_detect_vortices_dict_contract():
    ds = io.create_sample_Dataset(n_frames=3, rows=12, cols=12)
    vs = ds.piv.detect_vortices(criterion="gamma2", threshold=0.05)
    assert isinstance(vs, list)
    for v in vs:
        assert {"label", "center", "centroid", "area", "peak"} <= set(v.keys())
        row, col = v["center"]
        assert isinstance(row, int) and isinstance(col, int)
        assert 0 <= row < 12
        assert 0 <= col < 12
        assert isinstance(v["area"], int)
        assert v["area"] >= 1
