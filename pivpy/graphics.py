# -*- coding: utf-8 -*-
"""
Various plots

"""

import numpy as np
import matplotlib.pyplot as pl
import xarray as xr
    
def showf_ownclass(data, var=None, units=None, fig=None):
    """ 
    showf(data, var, units)
    """
    fig = pl.figure(None if fig is None else fig.number)
    # import pdb; pdb.set_trace()
    xlabel = (None if var is None else var[0]) + ' [' + (None if units is None else units[0])+']'
    ylabel = (None if var is None else var[1]) + ' [' + (None if units is None else units[1])+']'
    
    if not isinstance(data,list):
        tmp = []
        tmp.append(data)
        data = tmp
        
    
    for k,d in enumerate(data):
        pl.quiver(d['x'],d['y'],d['u'],d['v'],d['u']**2 + d['v']**2)
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        pl.title(str(k))
        pl.draw()
        pl.pause(0.1)
        
    pl.show()
        
         
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


    fig = pl.figure(None if fig is None else fig.number)  
    if 't' in data.dims:
        for t in data['t']:
            d = data.isel(t=t)
            pl.quiver(d['x'],d['y'],d['u'],d['v'],d['u']**2 + d['v']**2)
            pl.xlabel(xlabel)
            pl.ylabel(ylabel)
            pl.draw()
            pl.pause(0.1)
    else:
        pl.quiver(data['x'],data['y'],data['u'],data['v'],data['u']**2 + data['v']**2)
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        pl.draw()
        
    pl.show()  

def showscal(data):
    """ 
    showf(data, var, units)
    Arguments:
        data : xarray.DataSet that contains dimensions of t,x,y
               and a variable w (scalar)
    """
    # fig = pl.figure(None if fig is None else fig.number)
    # import pdb; pdb.set_trace()
    # xlabel = (None if var is None else var[0]) + ' [' + (None if units is None else units[0])+']'
    # ylabel = (None if var is None else var[1]) + ' [' + (None if units is None else units[1])+']'
        
    pl.figure()
    for t in data['t']:
        d = data.isel(t=t)
        pl.contour(d['x'],d['y'],d['w'])
        # pl.xlabel(xlabel)
        # pl.ylabel(ylabel)
        # pl.title(str(k))
        pl.draw()
        pl.pause(0.1)
        
    pl.show()              
     
