""" First example """

from pivpy.io import loadvec_dir
from pivpy.process import averf
from pivpy.graphics import showf

test_dir = "./examples/data"
data, var, units = loadvec_dir(test_dir)
mean = averf(data)
showf(mean,var,units)
