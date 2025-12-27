import importlib.resources

import numpy as np

from pivpy import io
import pivpy.pivpy  # noqa: F401 (register accessor)


try:
    from importlib.resources import files
except ImportError:
    from importlib.resources import path as resource_path


try:
    data_path = files("pivpy") / "data"
except NameError:
    with resource_path("pivpy", "data") as p:
        data_path = p


def test_batchf_bracket_expansion_and_callable():
    # Files Run000001..Run000003 exist in pivpy/data/Insight
    pat = str(
        data_path
        / "Insight"
        / "Run[1:3,6].T000.D000.P000.H001.L.vec"
    )

    def u_mean(ds):
        return float(np.nanmean(ds["u"].values))

    res = io.batchf(pat, u_mean)
    assert isinstance(res, list)
    assert len(res) == 3
    assert all(isinstance(v, float) for v in res)


def test_batchf_string_calls_accessor_method():
    # Call azprofile on each file; should return tuples.
    pat = str(
        data_path
        / "Insight"
        / "Run[1:2,6].T000.D000.P000.H001.L.vec"
    )
    res = io.batchf(pat, "azprofile", 0.0, 0.0, 1.0, 8)
    assert len(res) == 2
    a0, ur0, ut0 = res[0]
    assert a0.shape == (8,)
    assert ur0.shape[0] == 8
    assert ut0.shape[0] == 8
