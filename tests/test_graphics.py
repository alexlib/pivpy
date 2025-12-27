""" tests of pivpy.graphics module """
import pathlib
import importlib.resources
from pivpy import io, graphics, pivpy 

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

# Convert to pathlib.Path if not already
path = pathlib.Path(path)

filename = path / "Insight" / "Run000001.T000.D000.P000.H001.L.vec"

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


def test_streamplot():
    """ tests streamplot
    """
    graphics.streamplot(_d)
    _d.piv.streamplot()


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
    filename = path / "openpiv_vec" / "exp1_001_b.vec"
    print(filename, filename.exists())
    _d = io.load_vec(filename)
    _d.isel(t=0).piv.quiver() # notice the warning

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

def test_autocorrelation_plot():
    """tests autocorrelation plot
    """
    # Test with default variable 'u' and default spatial_average=True
    graphics.autocorrelation_plot(_d)
    
    # Test with variable 'v'
    graphics.autocorrelation_plot(_d, variable='v')
    
    # Test with spatial_average=False (flatten all dimensions)
    graphics.autocorrelation_plot(_d, variable='u', spatial_average=False)
    
    # Test with accessor
    _d.piv.autocorrelation_plot(variable='u')
    _d.piv.autocorrelation_plot(variable='v', spatial_average=False)


def test_histscal_disp():
    """tests histscal_disp"""
    d = io.create_sample_Dataset(n_frames=3, rows=8, cols=8)
    fig, ax = graphics.histscal_disp(d, variable='u', smooth=0, opt='n')
    assert fig is not None
    assert ax is not None

    # log-y option
    fig2, ax2 = graphics.histscal_disp(d, variable='u', smooth=0, opt='nl')
    assert ax2.get_yscale() == 'log'

    # accessor wrapper
    fig3, ax3 = d.piv.histscal_disp(variable='u', smooth=0, opt='n')
    assert fig3 is not None
    assert ax3 is not None


def test_histvec_disp():
    """tests histvec_disp"""
    d = io.create_sample_Dataset(n_frames=3, rows=8, cols=8)
    fig, ax = graphics.histvec_disp(d, smooth=0, opt='n')
    assert fig is not None
    assert ax is not None

    fig2, ax2 = graphics.histvec_disp(d, smooth=0, opt='nl')
    assert ax2.get_yscale() == 'log'

    fig3, ax3 = d.piv.histvec_disp(smooth=0, opt='n')
    assert fig3 is not None
    assert ax3 is not None