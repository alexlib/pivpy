# -*- coding: utf-8 -*-
"""
Various plots

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr
import os

def quiver(data, arrScale = 25.0, threshold = None, nthArr = 1, 
              contourLevels = None, colbar = True, logscale = False,
              aspectratio='equal', colbar_orient = 'vertical', units = None):
    """
    Generates a quiver plot of a 'data' xarray DataArray object (single frame from a dataset)
    Inputs:
        data - xarray DataArray of the type defined in pivpy, one of the frames in the Dataset
            selected by default using .isel(t=0)
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
        graphics.quiver(data, arrScale = 0.2, threshold = Inf, n)
    """
    
    data = dataset_to_array(data)
        
    x = data.x
    y = data.y
    u = data.u.T
    v = data.v.T
    
    if units is not None: # replace  units
        lUnits = units[0] # ['m' 'm' 'mm/s' 'mm/s']
        velUnits = units[2]
        tUnits = velUnits.split('/')[1] # make it 's' or 'dt'
    else:
        lUnits = data.attrs['units'][0]
        velUnits = data.attrs['units'][2]
        tUnits = data.attrs['units'][2].split('/')[-1]
    
    
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
        levels = np.linspace(0, np.max(S.flatten()), 30) # default contour levels up to max of S
    else:
        levels = np.linspace(0, contourLevels, 30)
                
    if logscale:
        c = ax.contourf(x,y,S,alpha=0.8,
                 cmap = plt.get_cmap("Blues"), 
                 levels = levels, norm = plt.colors.LogNorm())
    else:
        c = ax.contourf(x,y,S,alpha=0.8,
                 cmap = plt.get_cmap("Blues"), 
                 levels=levels)
    if colbar:
        cbar = plt.colorbar(c, orientation=colbar_orient)
        cbar.set_label(r'$ V \, (' + velUnits + r')$' )
        
    ax.quiver(x[::nthArr],y[::nthArr],
               u[::nthArr,::nthArr],v[::nthArr,::nthArr],units='width',
               scale = np.max(S*arrScale),headwidth=2)
    ax.set_xlabel('x (' + lUnits + ')')
    ax.set_ylabel('y (' + lUnits + ')')
    ax.set_aspect(aspectratio)       
    return fig,ax

def histogram(data, normed = False):
    """
    this function will plot a normalized histogram of
    the velocity data.
    Input:
        data : xarray DataSet with ['u','v'] attrs['units']
        normed : (optional) default is False to present normalized
        histogram
        
    """
    
    u = np.asarray(data.u).flatten()
    v = np.asarray(data.v).flatten()
    
    units = data.attrs['units']
    f,ax = plt.subplots(2)
    
    ax[0].hist(u,bins=np.int(np.sqrt(len(u))*0.5),density=normed)
    ax[0].set_xlabel('u ['+units[2]+']')
    
    ax[1] = plt.subplot2grid((2,1),(1,0))
    ax[1].hist(v,bins=np.int(np.sqrt(len(v)*0.5)),density=normed)
    ax[1].set_xlabel('v ['+units[2]+']')
    plt.tight_layout()
    return f, ax


def contour_plot(data, threshold = None, contourLevels = None, 
                    colbar = True,  logscale = False, aspectration='equal', units=None):
    """ contourf ajusted for the xarray PIV dataset, creates a 
        contour map for the data['w'] property. 
        Input:
            data : xarray PIV DataArray, converted automatically using .isel(t=0)
            threshold : a threshold value, default is None (no data clipping)
            contourLevels : number of contour levels, default is None
            colbar : boolean (default is True) show/hide colorbar 
            logscale : boolean (True is default) create in linear/log scale
            aspectration : string, 'equal' is the default
        
    """

    data = dataset_to_array(data)

    
    if units is not None:
        lUnits = units[0] # ['m' 'm' 'mm/s' 'mm/s']
        # velUnits = units[2]
        # tUnits = velUnits.split('/')[1] # make it 's' or 'dt'
    else:
        # lUnits, velUnits = '', ''
        lUnits = data.attrs['units'][0]
        
    f,ax = plt.subplots()    
    
    if threshold is not None:
        data['w'] = xr.where(data['w']>threshold, threshold, data['w'])
        
    m = np.amax(abs(data['w']))
    n = np.amin(abs(data['w']))
    if contourLevels == None:
        levels = np.linspace(np.min(data['w']),np.max(data['w']), 10)
    else:
        levels = contourLevels # vector of levels to set
        
    if logscale:
        c = ax.contourf(data.x,data.y,np.abs(data['w'].T), levels=levels,
                 cmap = plt.get_cmap('RdYlBu'), norm=plt.colors.LogNorm())
    else:
        c = ax.contourf(data.x,data.y,data['w'].T, levels=levels,
                 cmap = plt.get_cmap('RdYlBu'))
        
    plt.xlabel('x [' + lUnits + ']')
    plt.ylabel('y [' + lUnits + ']')
    if colbar:
        cbar = plt.colorbar(c)
        cbar.set_label(r'$\omega$ [s$^{-1}$]')
    ax.set_aspect(aspectration)
    return f,ax
        
         
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

def showscal(data, property='ke'):
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
    
    
    data = data.piv.vec2scal(property=property)
    contour_plot(data)                
        

def animate(data, arrowscale=1, savepath=None):
    """ animates the quiver plot for the dataset (multiple frames) 
    Input:
        data : xarray PIV type of DataSet
        arrowscale : [optional] integer, default is 1
        savepath : [optional] path to save the MP4 animation, default is None
    
    Output:
        if savepath is None, then only an image display of the animation
        if savepath is an existing path, a file named im.mp4 is saved
    
    """    
    X, Y = data.x, data.y
    U, V = data.u[:,:,0], data.v[:,:,0] # first frame
    fig, ax = plt.subplots(1,1)
    M = np.sqrt(U**2 + V**2)
    
    Q = ax.quiver(X[::3,::3], Y[::3,::3], 
                  U[::3,::3], V[::3,::3], M[::3,::3],
                 units='inches', scale=arrowscale)
    
    cb = plt.colorbar(Q)
    
    units = data.attrs['units']
    
    cb.ax.set_ylabel('velocity (' + units[2] + ')')
    
    text = ax.text(0.2,1.05, '1/'+str(len(data.t)), ha='center', va='center',
                   transform=ax.transAxes)
    
    def update_quiver(num,Q,data,text):
        U,V = data.u[:,:,num],data.v[:,:,num]
        
        M = np.sqrt(U[::3,::3]**2 + V[::3,::3]**2)   
        Q.set_UVC(U,V,M)
        text.set_text(str(num+1)+'/'+str(len(data.t)))
        return Q

    anim = FuncAnimation(fig, update_quiver, fargs=(Q,data,text),
                               frames = len(data.t), blit=False)
    mywriter = FFMpegWriter()
    if savepath:
        p = os.getcwd()
        os.chdir(savepath)
        anim.save('im.mp4', writer=mywriter)
        os.chdir(p)
    else: anim.save('im.mp4', writer=mywriter)  
    
    

def dataset_to_array(data,N=0):
    """ converts xarray Dataset to array """
    if 't' in data.dims:
        print('Warning: function for a single frame, using first frame, supply data.isel(t=N)')
        data = data.isel(t=N)
    return data