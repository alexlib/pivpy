=========================
PIVPy introduction
=========================

------------
History and motivation
------------


Federic Moisy created wonderful package, called PIVMAT http://www.fast.u-psud.fr/pivmat/ which 
is simpler than our GUI-based post-processing package, http://www.openpiv.net/openpiv-spatial-analysis-toolbox/

BUT, it's in Matlab which we decided to abandon and move all our activity to Python. So we also try to 
translate the PIVMAT into PIVPy. Of course, it won't be one-to-one translation because Python 
is by far richer language and provides us with some basic programming things that Matlab cannot. We 
hope that our effort will be useful. 



-------------
PIV data structure
-------------

Particle Image Velocimetry (PIV) data can be easily presented as a list of Numpy arrays, 
where each array has the datatype of `x,y,u,v,mask,quality`, where `x,y` are coordinates, 
'u,v' (and if data is 3D `x,y,z,u,v,w`, then also `w`)  

