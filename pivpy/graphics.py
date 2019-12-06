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
              aspectratio='equal',colbar = False, colbar_orient = 'vertical', 
              units = None, streamlines=False):
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

    if 'z' in data.dims:
        print('Warning: using first z cordinate')
        data = data.sel(z=0)   
    x = data.x
    y = data.y
    u = data.u.T
    v = data.v.T

    
    if units is not None: # replace  units
        lUnits = units[0] # ['m' 'm' 'mm/s' 'mm/s']
        velUnits = units[2]
        # tUnits = velUnits.split('/')[1] # make it 's' or 'dt'
    else:
        lUnits = data.attrs['units'][0]
        velUnits = data.attrs['units'][2]
        # tUnits = data.attrs['units'][2].split('/')[-1]
    
    
    if threshold is not None:
        data['u'] = xr.where(data['u']>threshold, threshold, data['u'])
        data['v'] = xr.where(data['v']>threshold, threshold, data['v'])
        
    S = np.array(np.sqrt(u**2 + v**2))
    
    if len(plt.get_fignums()) == 0: # if no figure is open
        fig, ax = plt.subplots() # open a new figure
    else:
        fig = plt.gcf()
        ax = plt.gca() 

    # quiver itself
    ax.quiver(x,y,u,v,units='width',
            scale = np.max(S*arrScale),headwidth=2)    
    
    if streamlines == True: # contours or streamlines
        speed = np.sqrt(u**2 + v**2)
        strm = ax.streamplot(x,y,u,v,color=speed,cmap=plt.get_cmap('hot'),linewidth=4) 

        if colbar:
            cbar = fig.colorbar(strm.lines, orientation=colbar_orient)
            cbar.set_label(r'$ V \, (' + velUnits + r')$' )
        
    
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
                    colbar = None, logscale = False, aspectratio='equal', 
                    units=None):
    """ contourf ajusted for the xarray PIV dataset, creates a 
        contour map for the data['w'] property. 
        Input:
            data : xarray PIV DataArray, converted automatically using .isel(t=0)
            threshold : a threshold value, default is None (no data clipping)
            contourLevels : number of contour levels, default is None
            colbar : None (hide), 'horizontal', or 'vertical' 
            logscale : boolean (True is default) create in linear/log scale
            aspectration : string, 'equal' is the default
        
    """

    if 'w' not in data.var():
        data.piv.vec2scal('ke')

    data = dataset_to_array(data)

    if units is not None:
        lUnits = units[0] # ['m' 'm' 'mm/s' 'mm/s']
        # velUnits = units[2]
        # tUnits = velUnits.split('/')[1] # make it 's' or 'dt'
    else:
        # lUnits, velUnits = '', ''
        lUnits = data.attrs['units'][0]
        propUnits = data.attrs['variables'][-1] + data.attrs['units'][-1] # last one is from 'w'
        
    f,ax = plt.subplots()    
    
    if threshold is not None:
        data['w'] = xr.where(data['w']>threshold, threshold, data['w'])
        
    m = np.amax(abs(data['w']))
    n = np.amin(abs(data['w']))
    if contourLevels == None:
        levels = np.linspace(np.min(data['w'].values), \
            np.max(data['w'].values), 10)
    else:
        levels = contourLevels # vector of levels to set
        
    if logscale:
        c = ax.contourf(data.x,data.y,np.abs(data['w'].T), levels=levels,
                 cmap = plt.get_cmap('RdYlBu'), norm=plt.colors.LogNorm())
    else:
        c = ax.contourf(data.x,data.y,data['w'].T, levels=levels,
                 cmap = plt.get_cmap('RdYlBu'))
        
    plt.xlabel(f'x [{lUnits}]')
    plt.ylabel(f'y [{lUnits}]')
    if colbar is not None:
        cbar = plt.colorbar(c,orientation=colbar)
        cbar.set_label(propUnits)

    ax.set_aspect(aspectratio)

    return f,ax
        
         
def showf(data, property='ke',**kwargs):
    """ 
    showf(data, var, units)
    Arguments:
        data : xarray.DataSet that contains dimensions of t,x,y
               and variables u,v and maybe w (scalar)
    """
    data.piv.vec2scal(property=property)
    contour_plot(data)
    quiver(data,**kwargs)


def showscal(data, property='ke',**kwargs):
    """ 
    showf(data, var, units)
    Arguments:
        data : xarray.DataSet that contains dimensions of t,x,y
               and a variable w (scalar)
    """
    data.piv.vec2scal(property=property)
    fig, ax = contour_plot(data,**kwargs)
    return fig, ax
        

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
    X, Y = np.meshgrid(data.x, data.y)
    X=X.T
    Y=Y.T
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
        Q.set_UVC(U[::3,::3],V[::3,::3],M[::3,::3])
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
    
    

def dataset_to_array(data,t=0):
    """ converts xarray Dataset to array """
    if 't' in data.dims:
        print('Warning: function for a single frame, using the first frame, supply data.isel(t=N)')
        data = data.isel(t=t)
    return data
