""" tests of pivpy.graphics module """
import pathlib
import pkg_resources as pkg
from pivpy import io, graphics, pivpy 

filename = pathlib.Path(
    pkg.resource_filename(
    "pivpy",  "data")) / "Insight" / "Run000001.T000.D000.P000.H001.L.vec"

# load data
_d = io.load_vec(filename).isel(t=0)


def test_showscal():
    """tests showscal
    """
    graphics.showscal(_d, flow_property="curl")


def test_quiver():
    """ tests quiver
    """
    graphics.quiver(_d)
    _d.piv.quiver()


def test_xarray_plot():
    """tests xarray plot use of pcolormesh 
    """
    d = _d.piv.vec2scal("curl")
    d["w"].plot.pcolormesh()


def test_histogram():
    """tests histogram
    """
    graphics.histogram(_d)


def test_quiver_openpiv_vec():
    """ tests quiver of openpiv vec file
    """
    filename = pathlib.Path(pkg.resource_filename("pivpy", "data")) / "openpiv" / "exp1_001_b.vec"
    print(filename, filename.exists())
    _d = io.load_vec(filename)
    _d.piv.quiver() # notice the warning

def test_showf():
    """tests showf
    """
    graphics.showf(_d)

def test_average():
    """tests average
    """
    d = io.create_sample_Dataset()
    d = d.piv.average
    d.piv.quiver()