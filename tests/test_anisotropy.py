from pivpy import io


def test_anisotropy_tensor():
    ds = io.create_sample_Dataset(n_frames=4, rows=8, cols=8)
    at = ds.piv.anisotropy_tensor()
    assert "b_xx" in at.data_vars and "b_xy" in at.data_vars


def test_anisotropy_invariants():
    ds = io.create_sample_Dataset(n_frames=4, rows=8, cols=8)
    inv = ds.piv.anisotropy_invariants()
    assert "neg_II" in inv.data_vars and "III" in inv.data_vars


def test_barycentric_map():
    ds = io.create_sample_Dataset(n_frames=4, rows=8, cols=8)
    bm = ds.piv.barycentric_map()
    assert "x_b" in bm.data_vars and "y_b" in bm.data_vars
