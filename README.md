[![PyPI version](https://badge.fury.io/py/pivpy.svg)](https://badge.fury.io/py/pivpy)
[![Documentation Status](https://readthedocs.org/projects/pivpy/badge/?version=latest)](https://pivpy.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alexlib/pivpy/master?filepath=examples%2Fnotebooks%2FGetting_Started.ipynb)



# PIVPy
Python based post-processing PIV data analysis, similar to PIVMAT and VecPy

Merging the three packages: 
1. https://github.com/tomerast/Vecpy
2. https://github.com/alexlib/pivpy/tree/xarray
3. https://github.com/ronshnapp/vecpy




### How do I get set up? ###

Use `pip`:  

    pip install pivpy
    
### How to get started? 

Look into the [getting started Jupyter notebook](https://github.com/alexlib/pivpy/blob/master/examples/notebooks/Getting_Started.ipynb)

and additional notebooks:
[Notebooks](https://github.com/alexlib/pivpy/blob/master/examples/notebooks/)



### How to test? ### 

From a command line just use:

    nosetests
    
### Documentation on Github:

[PIVPy on ReadTheDocs](http://pivpy.readthedocs.io)

### How to help? ###

Read the ToDo file and pick one item to program. Use Fork-Develop-Pull Request model to 
contribute

### How to write tutorials and add those to the documentation ###

Using great tutorial http://sphinx-ipynb.readthedocs.org/en/latest/howto.html we now can 
prepare IPython notebooks (see in /docs/source) and convert those to .rst files, then 

    python setup.py build_sphinx
    
generates ```html``` directory with the documentation ready
