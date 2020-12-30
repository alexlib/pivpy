from pivpy import io, graphics, pivpy
import pkg_resources as pkg

# f1 = 'Run000001.T000.D000.P000.H001.L.vec'
filename = pkg.resource_filename(
    "pivpy", "data/Insight/Run000001.T000.D000.P000.H001.L.vec"
)
# load data
_d = io.load_vec(filename)


def test_showscal():
    graphics.showscal(_d, property="ke")


def test_quiver():
    graphics.quiver(_d)
    _d.piv.quiver()


def test_xarray_plot():
    _d.piv.vec2scal(property="curl")
    _d["w"].isel(t=0).plot.pcolormesh()


def test_histogram():
    graphics.histogram(_d)


def test_showf():
    graphics.showf(_d)


def test_quiver_openpiv_vec():
    filename = pkg.resource_filename("pivpy", "data/openpiv/exp1_001_b.vec")
    # load data
    _d = io.load_vec(filename)
    _d.piv.quiver()
