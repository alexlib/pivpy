# -*- coding: utf-8 -*-

"""
Contains functions for reading flow fields in various formats
"""

import numpy as np
from glob import glob
import os.path


def loadvec_dir(directory):
    """ loadvec_dir (directory)
    """
    files  = glob(os.path.join(directory,'*.vec'))
    variables, units, rows, cols = parse_header(files[0])
    
    data = []
    for f in files:
        data.append(loadvec(f,rows,cols))
    
    return (data, variables, units) 

def parse_header(filename):
    """ parse_header (filename) """
    with open(filename) as fid:
        header = fid.readline()

    header_list = header.replace(',',' ').replace('=',' ').replace('"',' ').split()
    
    variables = header_list[3:12][::2]
    units = header_list[4:12][::2]
    rows = int(header_list[-5])
    cols = int(header_list[-3])
    
    
    return (variables, units, rows, cols)
    
        

def loadvec(filename,rows=None, cols=None):
    """
        loadvec(filename)
    """
    d = np.loadtxt(filename,skiprows=1,delimiter=',',
        dtype={'names': ('x', 'y', 'u', 'v'),'formats': ('f4', 'f4', 'f4', 'f4')},
        usecols=(0,1,2,3)).reshape(rows,cols)
    return d
    
    
    


