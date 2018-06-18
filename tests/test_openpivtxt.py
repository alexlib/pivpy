#!/usr/bin/python

from vecPy import loadVec
from vecPy import vecPlot
from vecPy.vecPy import Vec
import matplotlib.pyplot as plt


test_dir = "tests/data"
lst = loadVec.read_directory(test_dir,ext='txt')
data = loadVec.get_data_openpiv_txt(lst[0],test_dir)
# dt = loadvec.get_dt(lst[0],test_dir)
dt = 1.0 # there is no dt in OpenPIV files
x,y,u,v,chc = loadVec.vecToMatrix(data)
vec = Vec(x,y,u,v,chc,dt,lUnits='pix',tUnits = 'dt')


resolution = 1.0/71.96 #[mm/px]
# vec.rotate(0)
vec.scale(resolution)

plt.figure()
vecPlot.genQuiver(vec)
plt.show()

plt.figure()
vecPlot.genVorticityMap(vec)
plt.show()


test_dir = "tests/data"
lst = loadVec.read_directory(test_dir,ext='vec')
data = loadVec.get_data_openpiv_vec(lst[0],test_dir)
dt = 1.0 # there is no dt in OpenPIV files
x,y,u,v,chc = loadVec.vecToMatrix(data)
vec = Vec(x,y,u,v,chc,dt,lUnits='pix',tUnits = 'dt')


resolution = 1.0/71.96 #[mm/px]
# vec.rotate(0)
vec.scale(resolution)

plt.figure()
vecPlot.genQuiver(vec)
plt.show()

plt.figure()
vecPlot.genVorticityMap(vec)
plt.show()


