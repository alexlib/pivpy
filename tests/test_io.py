from pivpy import io, vecplot
import matplotlib.pyplot as plt

import os
fname = 'Run000001.T000.D000.P000.H001.L.vec'
path = './data/'


def test_parse_header():
    """ test if we get correct delta t """
    _, _, _, _, dt = io.parse_header(os.path.join(path,fname))
    assert dt == 2000.

def test_get_units():
    # test vec file with m/s
    lUnits,vUnits,tUnits = io.get_units(os.path.join(path,fname))
    assert lUnits == 'mm'
    assert vUnits == 'm/s'
    assert tUnits == 's'

    # test vec file with pixels/dt
    lUnits,vUnits,tUnits = io.get_units(os.path.join(path,'day2a005000.T000.D000.P003.H001.L.vec'))
    assert lUnits == 'pixel'
    assert vUnits == 'pixel/dt'
    assert tUnits == 'dt'

    # test OpenPIV vec 
    lUnits,vUnits,tUnits = io.get_units(os.path.join(path,'exp1_001_b.vec') )
    assert lUnits is 'pixel'

def test_loadvec():
    data = io.loadvec(os.path.join(path,fname))
    assert data['u'].shape == (63,63)
    assert data['u'][0,0] == 0.0
    assert data.coords['x'][0] == 0.31248