


#  <img src="pivpy_logo.png" alt="PIVPy" width="120" height="120">  PIVPy 

Python based post-processing PIV data analysis


[![PyPI version](https://badge.fury.io/py/pivpy.svg)](https://badge.fury.io/py/pivpy)
[![Documentation Status](https://readthedocs.org/projects/pivpy/badge/?version=latest)](https://pivpy.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alexlib/pivpy/master?filepath=examples%2Fnotebooks%2FGetting_Started.ipynb)



Merging the three packages: 
1. https://github.com/tomerast/Vecpy
2. https://github.com/alexlib/pivpy/tree/xarray
3. https://github.com/ronshnapp/vecpy




### How do I get set up? ###

Use `pip`:  

    pip install pivpy
    
### What packages are required and which are optional

1. `lvreader` by Lavision if you use vc7 files
2. `netcdf4` if you want to store NetCDF4 files by xarray
3. `pyarrow` if you want to store parquet files
4. `numpy`, `scipy`, `matplotlib`, `xarray` are must and installed with the `pivpy`
    
### How to get started? 

Look into the [getting started Jupyter notebook](https://github.com/alexlib/pivpy/blob/master/examples/notebooks/Getting_Started.ipynb)

and additional notebooks:
[Notebooks](https://github.com/alexlib/pivpy/blob/master/examples/notebooks/)



### How to test? ### 

From a command line just use:

    pip install pytest
    pytest
    
### Documentation on Github:

[PIVPy on ReadTheDocs](http://pivpy.readthedocs.io)

### How to help? ###

Read the ToDo file and pick one item to program. Use Fork-Develop-Pull Request model to 
contribute

### How to write tutorials and add those to the documentation ###

Using great tutorial http://sphinx-ipynb.readthedocs.org/en/latest/howto.html we now can 
prepare IPython notebooks (see in /docs/source) and convert those to .rst files, then 

    python setup.py sphinx-build
    sphinx-build -b html docs/source/ docs/build/html
    
generates ```docs/build/html``` directory with the documentation
