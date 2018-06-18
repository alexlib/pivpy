from pivpy import io, vecplot
from pivpy.pivpy import VectorField
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

def test_get_dt():
    """ test if we get correct delta t """
    import os

    fname = 'Run000001.T000.D000.P000.H001.L.vec'
    path = './data/'

    with open(os.path.join(os.path.abspath(path),fname)) as f:
        header = f.readline()
        
    ind1 = header.find('MicrosecondsPerDeltaT')
    dt = float(header[ind1:].split('"')[1])
    assert dt == 2000.