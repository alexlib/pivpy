import numpy as np
from pivpy import io, pivpy
import matplotlib.pyplot as plt

import os
import pkg_resources as pkg

path = pkg.resource_filename("pivpy", "data")
fname = os.path.join(path, "Insight", "Run000002.T000.D000.P000.H001.L.vec")

# def test_get_dt():
#     """ test if we get correct delta t """
#     _, _, _, _,dt,_ = io.parse_header(os.path.join(path,fname))
#     assert dt == 2000.


def test_get_frame():
    """ tests the correct frame number """
    _, _, _, _, _, frame = io.parse_header(
        os.path.join(path, "day2/day2a005003.T000.D000.P003.H001.L.vec")
    )
    assert frame == 5003
    _, _, _, _, _, frame = io.parse_header(
        os.path.join(path, "Insight/Run000002.T000.D000.P000.H001.L.vec")
    )
    assert frame == 2
    _, _, _, _, _, frame = io.parse_header(
        os.path.join(path, "openpiv/exp1_001_b.vec")
    )
    assert frame == 1
    _, _, _, _, _, frame = io.parse_header(
        os.path.join(path, "openpiv/exp1_001_b.txt")
    )
    assert frame == 1


def test_get_units():
    # test vec file with m/s
    lUnits, vUnits, tUnits = io.get_units(fname)
    assert lUnits == "mm"
    assert vUnits == "m/s"
    assert tUnits == "s"

    # test vec file with pixels/dt
    lUnits, vUnits, tUnits = io.get_units(
        os.path.join(path, "day2/day2a005000.T000.D000.P003.H001.L.vec")
    )
    assert lUnits == "pixel"
    assert vUnits == "pixel/dt"
    assert tUnits == "dt"

    # test OpenPIV vec
    lUnits, vUnits, tUnits = io.get_units(
        os.path.join(path, "openpiv/exp1_001_b.vec")
    )
    assert lUnits == "pix"


def test_load_vec():
    fname = "Run000001.T000.D000.P000.H001.L.vec"
    data = io.load_vec(os.path.join(path, "Insight", fname))
    assert data["u"].shape == (63, 63, 1)
    assert data["u"][0, 0, 0] == 0.0
    assert np.allclose(data.coords["x"][0], 0.31248)
    assert "t" in data.dims


def test_load_vc7():
    data = io.load_vc7(os.path.join(path, "VC7/2Ca.VC7"))
    assert data["u"].shape == (57, 43, 1)
    assert np.allclose(data.u.values[0, 0], -0.04354814)
    assert np.allclose(data.coords["x"][-1], 193.313795)


def test_loadopenpivtxt():
    data = io.load_vec(os.path.join(path, "openpiv", "exp1_001_b.txt"))


def test_load_directory():
    _ = pkg.resource_filename("pivpy", "data/Insight")
    data = io.load_directory(_, basename="Run*", ext=".vec")
    assert np.allclose(data["t"], [0, 1, 2, 3, 4])

    _ = pkg.resource_filename("pivpy", "data/VC7")
    data = io.load_directory(_, basename="2*", ext=".VC7")
    assert np.allclose(data["t"], [0, 1])

    data = io.load_directory(
        path=os.path.join(path, "urban_canopy"), basename="B*", ext=".vc7"
    )
    assert np.allclose(data["t"], [0, 1, 2, 3, 4])


def test_create_sample_field():
    data = io.create_sample_field(frame=3)
    assert data["t"] == 3
    data = io.create_sample_field(rows=3, cols=7)
    assert data.x.shape[0] == 7
    assert data.y.shape[0] == 3
    assert data["t"] == 0


def test_create_sample_dataset(n=3):
    data = io.create_sample_dataset(n=n)
    assert data.dims["t"] == 3
    assert np.allclose(data["t"], np.arange(3))
