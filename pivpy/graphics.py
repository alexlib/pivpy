# -*- coding: utf-8 -*-
"""
Various plots

"""

import numpy as np, matplotlib.pyplot as pl   
    
def showf(data, var=None, units=None, fig=None):
    """ 
    showf(data, var, units)
    """
    fig = pl.figure(None if fig is None else fig.number)
    pl.hold(False)
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
        
         
             
     
