# pivpy
Python based post-processing PIV data analysis, similar to PIVMAT and VecPy

Merging the three packages: 
1. https://github.com/tomerast/Vecpy
2. https://github.com/alexlib/pivpy/tree/xarray
3. https://github.com/ronshnapp/vecpy




### How do I get set up? ###

to start working just download the code and run the init.py script file.
this script imports all the other code needed for running the program properly. also, given a proper path destination and a file name of an example .vec file, this script will generate example plots.    


### How to test? ### 

From a command line just use:

    nosetests
    

### How to help? ###

Read the ToDo file and pick one item to program. Use Fork-Develop-Pull Request model to 
contribute

### How to write tutorials and add those to the documentation ###

Using great tutorial http://sphinx-ipynb.readthedocs.org/en/latest/howto.html we now can 
prepare IPython notebooks (see in /docs/source) and convert those to .rst files, then 

    python setup.py build_sphinx
    
generates ```html``` directory with the documentation ready