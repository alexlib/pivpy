# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:04:24 2015

@author: Ron

This module contains function that help generating
velocity attributes of the .vec files that were open.
use this code over vec objects.
"""

from numpy import linspace, amax, sqrt, gradient, cos, sin
from matplotlib.pyplot import colorbar, get_cmap, subplots, contourf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from os import chdir, getcwd
plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin/ffmpeg.exe'
from scipy.ndimage.filters import median_filter



def genQuiver(vec, arrScale = 25.0, threshold = None, nthArr = 1, 
              contourLevels = None, colbar = True, logscale = False,
              aspectratio='equal', colbar_orient = 'vertical'):
    """
    Generates a quiver plot of a 'vec' object
    Inputs:   
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
        vecplot.genQuiver(vec, arrScale = 0.2, threshold = Inf, n)
    """
    
    u = vec.u
    v = vec.v
    
    if threshold is not None:
        u = thresholdArray(u, threshold)
        v = thresholdArray(v, threshold)
        
    S = np.sqrt(u**2 + v**2)
    
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
        c = ax.contourf(vec.x,vec.y,S,alpha=0.8,
                 cmap = plt.get_cmap("Blues"), 
                 levels = levels, norm = colors.LogNorm())
    else:
        c = ax.contourf(vec.x,vec.y,S,alpha=0.8,
                 cmap = plt.get_cmap("Blues"), 
                 levels=levels)
    if colbar:
        cbar = plt.colorbar(c, orientation=colbar_orient)
        cbar.set_label(r'$\left| \, V \, \right|$ ['+vec.lUnits+' $\cdot$ '+vec.tUnits+'$^{-1}$]')
    n = nthArr
    ax.quiver(vec.x[1::n,1::n],vec.y[1::n,1::n],
               u[1::n,1::n],v[1::n,1::n],units='width',
               scale = np.max(S*arrScale),headwidth=2)
    ax.set_xlabel('x [' + vec.lUnits + ']')
    ax.set_ylabel('y [' + vec.lUnits + ']')
    ax.set_aspect(aspectratio)       
    return fig,ax
    
    


def quiverVecList(vec_list, shape = None, arrScale = 25.0, threshold = None,
                  nthArr = 1, contourLevels = None, colbar = True, 
                  logscale = False, aspectratio='equal'):
    '''
    will plot quiver plots from a list of vec instances, as subplots in 
    a sigle figure
    '''
    N = len(vec_list)
    if shape == None:
        rows = min(N, 3)
        cols = np.ceil(1.0*N/rows)
    elif type(shape) == tuple and len(shape)==2:
        row, cols = shape
    else:
        raise ValueError('bad input - shape is either tuple of None')
        
    fig = plt.figure() 
    ax = [fig.add_subplot(cols, rows, i) for i in range(N)]
        
    XY = vec_list[0].x, vec_list[0].y
    UV_list = [(i.u, i.v) for i in vec_list]
    S_list = [ np.sqrt(i[0]**2 + i[1]**2) for i in UV_list]
    
    S_max = np.amax( max(S_list, key=np.amax) )
        
    if contourLevels is None:
        levels = np.linspace(0, S_max, 30) # default contour levels up to max of S
    else:
        levels = np.linspace(0, contourLevels, 30)
    
    for i in range(N):
        a = ax[i]
        a.set_aspect(aspectratio) 
        if logscale:
            c = a.contourf(XY[0], XY[1], S_list[i], alpha=0.8,
                            cmap = plt.get_cmap("Blues"), 
                            levels = levels, norm = colors.LogNorm())
        else:
            c = a.contourf(XY[0], XY[1], S_list[i], alpha=0.8,
                            cmap = plt.get_cmap("Blues"), levels=levels)
        n = nthArr
        a.quiver(XY[0][1::n,1::n],XY[1][1::n,1::n],
               UV_list[i][0][1::n,1::n],UV_list[i][1][1::n,1::n],units='width',
               scale = S_max*arrScale, headwidth=2)
        a.set_xlabel('x [' + vec_list[0].lUnits + ']')
        a.set_ylabel('y [' + vec_list[0].lUnits + ']')
        a.set_aspect(aspectratio) 
    plt.tight_layout()
    return fig, ax



def genFluctuationQuiver(vec):
    """
    generate a quiver plot of velocity fluctuation
    i.e. velocity - mean_velocity.    
    """
    vec.getVelStat()
    u,v = vec.u-vec.Umean,vec.v-vec.Vmean
    S = sqrt(u**2+v**2)
    levels = linspace(0, amax(S), 30)
    contourf(vec.x,vec.y,S,alpha=0.5,
                 cmap = get_cmap("Greens"),
                 levels=levels)
    cbar = colorbar()
    cbar.set_label(r'Velocity [m $\cdot$ s$^{-1}$]')
    plt.quiver(vec.x, vec.y, u, v, units='width',
               scale=amax(sqrt(u**2+v**2))*25.0,headwidth=2 )
    plt.xlabel('x [' + vec.lUnits + ']')
    plt.ylabel('y [' + vec.lUnits + ']')
    
    
    
def genVelHist(vec, normed = False):
    """
    this function will plot a normalized histogram of
    the velocity data.
    """
    u1, v1 = vec.u.flatten(), vec.v.flatten()
    f,ax = subplots(2)
    ax1,ax2 = ax
    ax1.hist(u1,bins=int(sqrt(len(u1))*0.5),normed=normed)
    ax1.set_xlabel('u ['+vec.velUnits+']')
    ax2 = plt.subplot2grid((2,1),(1,0))
    ax2.hist(v1,bins=int(sqrt(len(v1)*0.5)),normed=normed)
    ax2.set_xlabel('v ['+vec.velUnits+']')
    plt.tight_layout()
    return f, ax1, ax2


    
def genVorticityMap(vec, threshold = None, contourLevels = None, 
                    colbar = True,  logscale = False, aspectration='equal'):
    """ why do we rotate the vector before taking derivative? """
    # BUG:
    dUy = gradient(vec.u)[0]*cos(vec.theta)-gradient(vec.u)[1]*sin(vec.theta)
    dVx = gradient(vec.v)[1]*cos(vec.theta)+gradient(vec.v)[0]*sin(vec.theta)
    dx = gradient(vec.x)[1]*cos(vec.theta)+gradient(vec.x)[0]*sin(vec.theta)
    dy = gradient(vec.y)[0]*cos(vec.theta)-gradient(vec.y)[1]*sin(vec.theta)
    vorticity = dVx/dy-dUy/dx
    
    f,ax = subplots()    
    
    if threshold != None:
        vorticity = thresholdArray(vorticity,threshold)
    m = amax(abs(vorticity))
    if contourLevels == None:
        levels = linspace(-m, m, 30)
    else:
        levels = linspace(-contourLevels, contourLevels, 30)
        
    if logscale:
        c = ax.contourf(vec.x,vec.y,np.abs(vorticity), levels=levels,
                 cmap = get_cmap('RdYlBu'), norm=colors.LogNorm())
    else:
        c = ax.contourf(vec.x,vec.y,vorticity, levels=levels,
                 cmap = get_cmap('RdYlBu'))
    plt.xlabel('x [' + vec.lUnits + ']')
    plt.ylabel('y [' + vec.lUnits + ']')
    if colbar:
        cbar = colorbar(c)
        cbar.set_label(r'$\omega$ [s$^{-1}$]')
    ax.set_aspect(aspectration)
    return f,ax


def genShearMap(vec, threshold = None, contourLevels = None, logscale = False,
                colbar = True, aspectratio='equal'):
    """this function plots a map of the xy strain e_xy"""
    dUy = gradient(vec.u)[0]*cos(vec.theta)-gradient(vec.u)[1]*sin(vec.theta)
    dVx = gradient(vec.v)[1]*cos(vec.theta)+gradient(vec.v)[0]*sin(vec.theta)
    dx = gradient(vec.x)[1]*cos(vec.theta)+gradient(vec.x)[0]*sin(vec.theta)
    dy = gradient(vec.y)[0]*cos(vec.theta)-gradient(vec.y)[1]*sin(vec.theta)
    strain = dVx/dy+dUy/dx
    
    f, ax = subplots()    
    
    if threshold != None:
        strain = thresholdArray(strain,threshold)
    m = amax(abs(strain))
    if contourLevels == None:
        levels = linspace(-m, m, 30)
    else:
        levels = linspace(-contourLevels, contourLevels, 30)
        
    if logscale:
        c = ax.contourf(vec.x,vec.y,np.abs(strain), levels=levels,
                 cmap = get_cmap('PRGn'), norm = colors.LogNorm())
    else:
        c = ax.contourf(vec.x,vec.y,strain, levels=levels,
                 cmap = get_cmap('PRGn'))
    plt.xlabel('x [' + vec.lUnits + ']')
    plt.ylabel('y [' + vec.lUnits + ']')
    if colbar:
        cbar = colorbar(c)
        cbar.set_label(r'$\epsilon_t$ [s$^{-1}$]')
    ax.set_aspect(aspectratio)
    return f, ax


def genFlowAcceleration(vec, arrScale = 25.0, threshold = None, nthArr = 1,
                        contourLevels = None, logscale = False, 
                        colbar=True, aspectratio='equal'):
    """this function will plot a contour plot of
    the convective term of material derivative.
    i.e. it plots the magnitude of the vector 
    (u*dudx + v*dudy , u*dvdx + v*dvdy)"""  
    dUx = gradient(vec.u)[1]*cos(vec.theta)+gradient(vec.u)[0]*sin(vec.theta)
    dUy = gradient(vec.u)[0]*cos(vec.theta)-gradient(vec.u)[1]*sin(vec.theta)
    dVx = gradient(vec.v)[1]*cos(vec.theta)+gradient(vec.v)[0]*sin(vec.theta)
    dVy = gradient(vec.v)[0]*cos(vec.theta)-gradient(vec.v)[1]*sin(vec.theta)
    dx = gradient(vec.x)[1]*cos(vec.theta)+gradient(vec.x)[0]*sin(vec.theta)
    dy = gradient(vec.y)[0]*cos(vec.theta)-gradient(vec.y)[1]*sin(vec.theta)
    ax = median_filter(vec.u*dUx/dx + vec.v*dUy/dy , (3,3))
    ay = median_filter(vec.u*dVx/dx + vec.v*dVy/dy , (3,3))
    if threshold != None:
        ax = thresholdArray(ax,threshold)
        ay = thresholdArray(ay,threshold)
    S = sqrt(ax**2+ay**2)
    if contourLevels == None:
        levels = linspace(0, amax(S), 30)
    else:
        levels = linspace(0, contourLevels, 30)
    f,axs = subplots()
    if logscale:
        c = axs.contourf(vec.x,vec.y,S,alpha=0.5,
                 cmap = get_cmap("OrRd"), 
                 levels=levels, norm=colors.LogNorm())
    else:
        c = axs.contourf(vec.x,vec.y,S,alpha=0.5,
                 cmap = get_cmap("OrRd"), 
                 levels=levels)
    if colbar:
        cbar = colorbar(c)
        cbar.set_label(r'$\left| \, \left( V\cdot \nabla \right) \cdot V \, \right|$ ['+
                   vec.lUnits+' $\cdot$ '+vec.tUnits+'$^{-2}$]')
    n = nthArr
    axs.quiver(vec.x[1::n,1::n],vec.y[1::n,1::n],
               ax[1::n,1::n],ay[1::n,1::n],units='width',
               scale=amax(S)*arrScale,headwidth=2 )
    axs.set_xlabel('x [' + vec.lUnits + ']')
    axs.set_ylabel('y [' + vec.lUnits + ']')
    axs.set_aspect(aspectratio)
    return f,axs


def animateVecList(vecList, arrowscale=1, savepath=None):
    X, Y = vecList[0].x, vecList[0].y
    U, V = vecList[0].u, vecList[0].v
    fig, ax = subplots(1,1)
    #Q = ax.quiver(X, Y, U, V, units='inches', scale=arrowscale)
    M = sqrt(pow(U, 2) + pow(V, 2))    
    Q = ax.quiver(X[::3,::3], Y[::3,::3], 
                  U[::3,::3], V[::3,::3], M[::3,::3],
                 units='inches', scale=arrowscale)
    cb = colorbar(Q)
    cb.ax.set_ylabel('velocity ['+vecList[0].lUnits+'/'+vecList[0].tUnits+']')
    text = ax.text(0.2,1.05, '1/'+str(len(vecList)), ha='center', va='center',
                   transform=ax.transAxes)
    def update_quiver(num,Q,vecList,text):
        U,V = vecList[num].u[::3,::3],vecList[num].v[::3,::3]
        M = sqrt(pow(U, 2) + pow(V, 2))   
        Q.set_UVC(U,V,M)
        #Q.set_UVC(vecList[num].u,vecList[num].v)
        text.set_text(str(num+1)+'/'+str(len(vecList)))
        return Q,
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q,vecList,text),
                               frames = len(vecList), blit=False)
    mywriter = animation.FFMpegWriter()
    if savepath:
        p = getcwd()
        chdir(savepath)
        anim.save('im.mp4', writer=mywriter)
        chdir(p)
    else: anim.save('im.mp4', writer=mywriter)  
    
    
    
def thresholdArray(array, th):
    index = np.where(abs(array)>th)
    for i in range(len(index[0])):
        array[index[0][i],index[1][i]] = th*np.sign(array[index[0][i],index[1][i]])
    return array
    

    
    