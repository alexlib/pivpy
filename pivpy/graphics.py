# -*- coding: utf-8 -*-
"""
Various plots

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def quiver(vec, arrScale = 25.0, threshold = None, nthArr = 1, 
              contourLevels = None, colbar = True, logscale = False,
              aspectratio='equal', colbar_orient = 'vertical', units = None):
    """
    Generates a quiver plot of a 'vec' xarray DataArray object (single frame from a dataset)
    Inputs:
        vec - xarray DataArray of the type defined in pivpy (u,v with coords x,y,t and the attributes)
        threshold - values above the threshold will be set equal to threshold
        arrScale - use to change arrow scales
        nthArr - use to plot only every nth arrow from the array 
        contourLevels - use to specify the maximum value (abs) of contour plots 
        colbar - True/False wether to generate a colorbar or not
        logscale - if true then colorbar is on log scale
        aspectratio - set auto or equal for the plot's apearence
        colbar_orient - 'horizontal' or 'vertical' orientation of the colorbar (if colbar is True)
    Outputs:
        none
    Usage:
        graphics.quiver(vec, arrScale = 0.2, threshold = Inf, n)
    """
    x = vec.x
    y = vec.y
    u = vec.u
    v = vec.v
    
    if units is not None:
        lUnits = units[0] # ['m' 'm' 'mm/s' 'mm/s']
        velUnits = units[2]
        tUnits = velUnits.split('/')[1] # make it 's' or 'dt'
    else:
        lUnits, velUnits, tUnits = '', '', ''
    
    
    if threshold is not None:
        data['u'] = xr.where(data['u']>threshold, threshold, data['u'])
        data['v'] = xr.where(data['v']>threshold, threshold, data['v'])
        
    S = np.array(np.sqrt(u**2 + v**2))
    
    fig = plt.get_fignums()
    if len(fig) == 0: # if no figure is open
        fig, ax = plt.subplots() # open a new figure
    else:
        ax = plt.gca()     
    
    if contourLevels is None:
        levels = np.linspace(0, np.max(S), 30) # default contour levels up to max of S
    else:
        levels = np.linspace(0, contourLevels, 30)
    if logscale:
        c = ax.contourf(x,y,S,alpha=0.8,
                 cmap = plt.get_cmap("Blues"), 
                 levels = levels, norm = colors.LogNorm())
    else:
        c = ax.contourf(x,y,S,alpha=0.8,
                 cmap = plt.get_cmap("Blues"), 
                 levels=levels)
    if colbar:
        cbar = plt.colorbar(c, orientation=colbar_orient)
        cbar.set_label(r'$\left| \, V \, \right|$ ['+ lUnits +' $\cdot$ '+ tUnits +'$^{-1}$]')
        
    ax.quiver(x[::nthArr],y[::nthArr],
               u[::nthArr,::nthArr],v[::nthArr,::nthArr],units='width',
               scale = np.max(S*arrScale),headwidth=2)
    ax.set_xlabel('x (' + lUnits + ')')
    ax.set_ylabel('y (' + lUnits + ')')
    ax.set_aspect(aspectratio)       
    return fig,ax
        
         
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


    for t in data['t']:
        d = data.isel(t=t)
        plt.quiver(d['x'],d['y'],d['u'],d['v'],d['u']**2 + d['v']**2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()
        plt.pause(0.1)
        
    plt.show()  

def showscal(data, property='ken'):
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
    
    d = process.vec2scal(data,property=property)

    plt.figure()
    for t in d['t']:
        tmp = d.isel(t=t)
        plt.contour(tmp['x'],tmp['y'],tmp['w'])
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.title(str(k))
        plt.draw()
        plt.pause(0.1)
    plt.show()              
     

        
def contourf(vec, threshold = None, contourLevels = None, 
                    colbar = True,  logscale = False, aspectration='equal'):
    """ contourf ajusted for the xarray PIV dataset, creates a 
        contour map for the data['w'] property. 
        Input:
            data : xarray PIV DataArray
            threshold : a threshold value, default is None (no data clipping)
            contourLevels : number of contour levels, default is None
            colbar : boolean (default is True) show/hide colorbar 
            logscale : boolean (True is default) create in linear/log scale
            aspectration : string, 'equal' is the default
        
    """
    
    if units is not None:
        lUnits = units[0] # ['m' 'm' 'mm/s' 'mm/s']
        velUnits = units[2]
        tUnits = velUnits.split('/')[1] # make it 's' or 'dt'
    else:
        lUnits, velUnits, tUnits = '', '', ''
        
    f,ax = subplots()    
    
    if threshold is not None:
        data['w'] = xr.where(data['w']>threshold, threshold, data['w'])
        
    m = np.amax(abs(data['w']))
    if contourLevels == None:
        levels = np.linspace(-m, m, 30)
    else:
        levels = np.linspace(-contourLevels, contourLevels, 30)
        
    if logscale:
        c = ax.contourf(vec.x,vec.y,np.abs(data['w']), levels=levels,
                 cmap = get_cmap('RdYlBu'), norm=colors.LogNorm())
    else:
        c = ax.contourf(vec.x,vec.y,data['w'], levels=levels,
                 cmap = get_cmap('RdYlBu'))
        
    plt.xlabel('x [' + lUnits + ']')
    plt.ylabel('y [' + lUnits + ']')
    if colbar:
        cbar = colorbar(c)
        cbar.set_label(r'$\omega$ [s$^{-1}$]')
    ax.set_aspect(aspectration)
    return f,ax