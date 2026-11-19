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
