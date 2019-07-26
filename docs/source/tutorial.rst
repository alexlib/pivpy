===================
Tutorial
===================


1. Download the package from the Github repository::

	git clone git@github.com:alexliberzonlab/pivpy.git

2. From the command line run the ``tmp.py``::

	% python tmp.py
		
or use the following code from your Python environment::

	from pivpy.io import load_vec_dir
	from pivpy.process import averf
	from pivpy.graphics import showf

	test_dir = "./examples/data"
	data, var, units = load_vec_dir(test_dir)
	mean = averf(data)
	showf(mean,var,units) 
	
	
And the result should look like: 

.. image:: out.png
	
