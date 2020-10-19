============
PIVPy basics
============

------------------
PIV data structure
------------------

Particle Image Velocimetry (PIV) data can be easily presented as a list of Numpy arrays, 
where each array has the datatype of ``x,y,u,v,mask,quality``, where ``x,y`` are coordinates, 
'u,v' (and if data is 3D ``x,y,z,u,v,w``, then also `w`)  

