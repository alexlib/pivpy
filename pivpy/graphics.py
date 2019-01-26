# -*- coding: utf-8 -*-
"""
Various plots

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from . import process
    
def showf_ownclass(data, var=None, units=None, fig=None):
    """ 
    showf(data, var, units)
    """
    fig = plt.figure(None if fig is None else fig.number)
    # import pdb; pdb.set_trace()
    xlabel = (None if var is None else var[0]) + ' [' + (None if units is None else units[0])+']'
    ylabel = (None if var is None else var[1]) + ' [' + (None if units is None else units[1])+']'
    
    if not isinstance(data,list):
        tmp = []
        tmp.append(data)
        data = tmp
        
    
    for k,d in enumerate(data):
        plt.quiver(d['x'],d['y'],d['u'],d['v'],d['u']**2 + d['v']**2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(str(k))
        plt.draw()
        plt.pause(0.1)
        
    plt.show()
        
         
def showf(data, variables=None, units=None, fig=None):
    """ 
    showf(data, var, units)
    Arguments:
        data : xarray.DataSet that contains dimensions of t,x,y
               and variables u,v and maybe w (scalar)
    """

    if variables is None:
        xlabel = ' '
        ylabel = ' '
    else:
        xlabel = variables[0]
        ylabel = variables[1]

    if units is not None:
        xlabel += ' ' + units[0]
        ylabel += ' ' + units[1]


    fig = plt.figure(None if fig is None else fig.number)  
    if 't' in data.dims:
        for t in data['t']:
            d = data.isel(t=t)
            plt.quiver(d['x'],d['y'],d['u'],d['v'],d['u']**2 + d['v']**2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.draw()
            plt.pause(0.1)
    else:
        plt.quiver(data['x'],data['y'],data['u'],data['v'],data['u']**2 + data['v']**2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()
        
    plt.show()  

def showscal(data,bckgr='ken'):
    """ 
    showf(data, var, units)
    Arguments:
        data : xarray.DataSet that contains dimensions of t,x,y
               and a variable w (scalar)
    """
    # fig = plt.figure(None if fig is None else fig.number)
    # import pdb; pdb.set_trace()
    # xlabel = (None if var is None else var[0]) + ' [' + (None if units is None else units[0])+']'
    # ylabel = (None if var is None else var[1]) + ' [' + (None if units is None else units[1])+']'
    
    d = process.vec2scal(data,property=bckgr)

    plt.figure()
    if 't' in data.dims: 
        for t in data['t']:
            d = data.isel(t=t)
            plt.contour(d['x'],d['y'],d['w'])
            # plt.xlabel(xlabel)
            # plt.ylabel(ylabel)
            # plt.title(str(k))
            plt.draw()
            plt.pause(0.1)
    else:
        plt.contour(data['x'],data['y'],data['w'])
        plt.draw()
    plt.show()              
     
