from pivpy import io, graphics, pivpy
import pkg_resources as pkg
import pathlib 

# f1 = 'Run000001.T000.D000.P000.H001.L.vec'
filename = pathlib.Path(
    pkg.resource_filename(
    "pivpy",  "data")) / "Insight" / "Run000001.T000.D000.P000.H001.L.vec"
# load data
_d = io.load_vec(filename).isel(t=0)


def test_showscal():
    graphics.showscal(_d, property="curl")


def test_quiver():
    graphics.quiver(_d)
    _d.piv.quiver()


def test_xarray_plot():
    d = _d.piv.vec2scal("curl")
    d["w"].plot.pcolormesh()


def test_histogram():
    graphics.histogram(_d)


def test_quiver_openpiv_vec():
    filename = pathlib.Path(pkg.resource_filename("pivpy", "data")) / "openpiv" / "exp1_001_b.vec"
    print(filename, filename.exists())
    _d = io.load_vec(filename)
    _d.piv.quiver() # notice the warning


# def test_showf():
#     graphics.showf(_d)