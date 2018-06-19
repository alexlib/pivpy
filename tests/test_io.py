from pivpy import io, vecplot
import matplotlib.pyplot as plt


# test_dir = "tests/data"
# # lst = loadVec.read_directory(test_dir)
# data = loadVec.get_data(lst[3],test_dir)
# dt = loadVec.get_dt(lst[3],test_dir)
# x,y,u,v,chc = loadVec.vecToMatrix(data)
# vec = Vec(x,y,u,v,chc,dt,lUnits='mm',tUnits = 's')


# resolution = 1.0/71.96 #[mm/px]
# vec.rotate(-90)
# vec.scale(resolution)

# plt.figure()
# vecPlot.genQuiver(vec)
# plt.show()

# plt.figure()
# vecPlot.genVorticityMap(vec)
# plt.show()


import os
fname = 'Run000001.T000.D000.P000.H001.L.vec'
path = './data/'


def test_get_dt():
    """ test if we get correct delta t """
    dt = io.get_dt(fname, path)
    assert dt == 2000.

def test_get_units():
    lUnits,vUnits,tUnits = io.get_units(fname, path)
    assert lUnits == 'mm'
    assert vUnits == 'm/s'
    assert tUnits == 's'

    lUnits,vUnits,tUnits = io.get_units('exp1_001_b.vec', path)
    assert lUnits is None

def test_loadvec():
    data = io.loadvec(os.path.join(path,fname))
    assert data['u'].shape == (63,63)
    assert data['u'][0,0] == 0.0
    assert data.coords['x'][0] == 0.31248

def test_vel_units():
    data = io.loadvec(os.path.join(path,fname))
    assert data.vel_units == 'm/s'