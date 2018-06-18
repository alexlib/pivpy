#!/usr/bin/python

from vecPy import loadVec
from vecPy import vecPlot
from vecPy.vecPy import Vec
import matplotlib.pyplot as plt


test_dir = "tests/data"
lst = loadVec.read_directory(test_dir)
data = loadVec.get_data(lst[3],test_dir)
dt = loadVec.get_dt(lst[3],test_dir)
x,y,u,v,chc = loadVec.vecToMatrix(data)
vec = Vec(x,y,u,v,chc,dt,lUnits='mm',tUnits = 's')


resolution = 1.0/71.96 #[mm/px]
vec.rotate(-90)
vec.scale(resolution)

plt.figure()
vecPlot.genQuiver(vec)
plt.show()

plt.figure()
vecPlot.genVorticityMap(vec)
plt.show()

