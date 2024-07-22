import pathlib
import numpy as np
import importlib.resources
from pivpy import io

# Ensure compatibility with different Python versions (3.9+ has 'files', 3.7 and 3.8 need 'path')
try:
    from importlib.resources import files
except ImportError:
    from importlib.resources import path as resource_path

# For Python 3.9+
try:
    path = files('pivpy') / 'data'
except NameError:
    # For Python 3.7 and 3.8
    with resource_path('pivpy', 'data') as data_path:
        path = data_path


vec_file = path / "Insight" / "Run000002.T000.D000.P000.H001.L.vec" 
openpiv_txt_file = path / "openpiv_txt" / "exp1_001_b.txt"


def test_get_dt():
    """ test if we get correct delta t """
    _, _, _, _,delta_t,_,_ = io.parse_header(vec_file)
    assert delta_t == 2000.


def test_get_frame():
    """ tests the correct frame number """
    _, _, _, _, _, frame,_ = io.parse_header(
        path/  "day2" / "day2a005003.T000.D000.P003.H001.L.vec"
    )
    assert frame == 5003
    _, _, _, _, _, frame,_ = io.parse_header(
        vec_file
    )
    assert frame == 2
    _, _, _, _, _, frame,_ = io.parse_header(
        path / "openpiv_vec" / "exp1_001_b.vec"
        )
    assert frame == 1
    _, _, _, _, _, frame,_ = io.parse_header(
        path / "openpiv_txt" / "exp1_001_b.txt"
    )
    assert frame == 1


def test_load_vec():
    """tests loading vec file
    """
    data = io.load_vec(vec_file)
    assert data["u"].shape == (63, 63, 1)
    tmp = data["u"].values
    assert tmp[0, 0, 0] == 0.0
    assert np.allclose(data.coords["x"][0], 0.31248)
    assert "t" in data.dims


# readim is depreceated, see the new Lavision Python package
# def test_load_vc7():
#     data = io.load_vc7(os.path.join(path, "VC7/2Ca.VC7"))
#     assert data["u"].shape == (57, 43, 1)
#     assert np.allclose(data.u.values[0, 0], -0.04354814)
#     assert np.allclose(data.coords["x"][-1], 193.313795)


def test_loadopenpivtxt():
    """tests loading openpivtxt file
    """
    io.load_openpiv_txt(openpiv_txt_file)

def test_load_directory():
    """tests loading directory of Insight VEC, vc7, and Davis8 files
    """
    data = io.load_directory(
        path / "Insight", 
        basename="Run*", 
        ext=".vec"
        )
    print(data.t)
    assert np.allclose(data["t"], [0, 1, 2, 3, 4])

    data = io.load_directory(
        path / "urban_canopy", 
        basename="B*", 
        ext=".vc7"
    )
    assert np.allclose(data["t"], [0, 1, 2, 3, 4])

    data = io.load_directory(
        path / "PIV_Challenge", 
        basename="B*", 
        ext=".txt"
    )
    assert np.allclose(data["t"], [0, 1])

def test_check_units():
    """ reads units and checks their validitty 
    def set_default_attrs(dataset: xr.Dataset)-> xr.Dataset:
    """ 
    data = io.create_sample_Dataset()
    assert data.t.attrs["units"] in ["s", "sec", "frame"]
    assert data.x.attrs["units"] in ["pix", "m", "mm"]
    assert data.y.attrs["units"] in ["pix", "m", "mm"]
    assert data.u.attrs["units"] in ["pix", "m", "mm"]
    assert data.v.attrs["units"] in ["pix", "m", "mm"]
    assert data.attrs["delta_t"] == 0.0


def test_create_sample_field():
    data = io.create_sample_field(frame=3)
    assert data["t"] == 3
    data = io.create_sample_field(rows=3, cols=7)
    assert data.x.shape[0] == 7
    assert data.y.shape[0] == 3
    assert data["t"] == 0.0
    


def test_create_sample_dataset():
    data = io.create_sample_Dataset(n_frames=3)
    assert data.sizes["t"] == 3
    # assert data.dims["t"] == 3
    assert np.allclose(data["t"], np.arange(3))


def test_to_nc():
    data = io.create_sample_Dataset(n_frames = 25)
    data.to_netcdf("tmp.nc")

    data = io.load_directory(path / "Insight" )
    data.to_netcdf("tmp.nc")

