
""" from vecPy import load_vec
from vecPy import vecPlot
from vecPy.vecPy import Vec
import matplotlib.pyplot as plt


test_dir = "tests/data"
lst = load_vec.read_directory(test_dir,ext='txt')
data = load_vec.get_data_openpiv_txt(lst[0],test_dir)
# dt = load_vec.get_dt(lst[0],test_dir)
dt = 1.0 # there is no dt in OpenPIV files
x,y,u,v,chc = load_vec.vecToMatrix(data)
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
lst = load_vec.read_directory(test_dir,ext='vec')
data = load_vec.get_data_openpiv_vec(lst[0],test_dir)
dt = 1.0 # there is no dt in OpenPIV files
x,y,u,v,chc = load_vec.vecToMatrix(data)
vec = Vec(x,y,u,v,chc,dt,lUnits='pix',tUnits = 'dt')


resolution = 1.0/71.96 #[mm/px]
# vec.rotate(0)
vec.scale(resolution)

plt.figure()
vecPlot.genQuiver(vec)
plt.show()

plt.figure()
vecPlot.genVorticityMap(vec)
plt.show() """


