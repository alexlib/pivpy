import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pivpy",
    version = "0.0.05",
    author = "Turbulence Structure Laboratory, Tel Aviv University",
    author_email = "alex.liberzon@gmail.com",
    description = ("Post processing of PIV vector files "
                                   " eventually going into OpenPIV "),
    license = "BSD",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/pivpy",
    packages=find_packages(),
    test_suite="tests",
    install_requires=[
        'numpy',
        'scipy',
        'xarray',
        'matplotlib',
        'readim',
        'pytest'
    ], 
    # packages=['pivpy', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
