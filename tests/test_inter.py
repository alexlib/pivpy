""" tests of pivpy.inter module """
import pathlib
import pkg_resources as pkg
import numpy as np
from pivpy import io, inter

path = pathlib.Path(pkg.resource_filename("pivpy", "data"))

openpivTxtTestFile = path / "openpiv_txt" / "interTest.txt"
saveNcFile  = path / "interTest" / "testInterCreates_nc.nc"

def test_pivpyTOvf():
    """
    The idea is to check if VortexFitting gets the same velocity field
    as PIVPY. But note that VortexFitting modifies the field a bit
    when reading it in.
    IMPORTANT: to run this test, VortexFitting package must be installed.
    """
    d = io.load_openpiv_txt(str(openpivTxtTestFile)) # PIVPY velocity field
    vfField = inter.pivpyTOvf(d, saveNcFile) # VortexFitting velocity field
    x = d.coords['x'].values
    y = d.coords['y'].values
    u = d['u'].isel(t=0).values
    v = d['v'].isel(t=0).values
    # See classes.py, below line 87 (if file_type == "piv_netcdf") in VortexFitting package.
    assert x.shape == vfField.x_coordinate_matrix.shape
    assert y.shape == vfField.y_coordinate_matrix.shape
    assert x.all() == vfField.x_coordinate_matrix.all()
    assert np.subtract(np.flip(y),np.flip(y)[0]).all() == vfField.y_coordinate_matrix.all()
    # Due to image to physical coordinate system converstion, u and v must be transposed.
    assert u.T.shape == vfField.u_velocity_matrix.shape
    assert v.T.shape == vfField.v_velocity_matrix.shape
    assert np.subtract(u.T,np.mean(u.T,1)[:, None]).all() == vfField.u_velocity_matrix.all()
    assert np.subtract(v.T,np.mean(v.T,1)[:, None]).all() == vfField.v_velocity_matrix.all()