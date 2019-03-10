import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


#General
def quiver(X,Y,U,V,title,units):
    Q = plt.figure()
    plt.title(title,size=30)
    ax1 = plt.quiver(X,Y,U,V,np.sqrt(U**2+V**2),angles='xy',cmap='coolwarm',scale=1e2,width=0.0009,headwidth=6,headlength=7)
    qk = plt.quiverkey(ax1, 0.9, 0.9, 1, r'1 '+units, labelpos='E',coordinates='figure')
    plt.colorbar()
    plt.xlabel('meter',size=20)
    plt.ylabel('meter',size=20)
    plt.show()
    return Q

def quiver_profile(grid_points,vel,title,units):
    plt.figure()
    plt.title(title,size=20)
    second_axis = np.zeros(grid_points.shape)
    Q = plt.quiver(second_axis,grid_points,vel,vel,width=0.0023,headwidth=3,headlength=5,angles='xy')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'1 '+units, labelpos='E',coordinates='figure')
    plt.show()
    
    
def contour(X,Y,Values,title):
    plt.figure()
    plt.title(title,size=20)
    cont = plt.contourf(X,Y,Values)
    plt.colorbar()
    plt.xlabel('')
    plt.ylabel('')
    plt.show()
    return cont

def plot(Data,grid,title,labels):
    #plt.figure()
    plt.title(title,size=20)
    plt.plot(Data , grid, 'o-' )
    plt.xlabel(labels[0],size=15)
    plt.ylabel(labels[1],size=15)
    plt.show()

#Frame
def frame_quiver(frame):
    X,Y = frame.return_grid()
    U,V = frame.return_all_velocities()
    title = 'Quiver of frame: '+ frame.properties.frame
    units = frame.properties.velocity_units
    quiver(X,Y,U,V,title,units)
    

#Run
def run_quiver_animation(run):
    fig, ax = plt.subplots(figsize=(21, 18))
    if run.check_same_grid_run():
        X,Y = run.run_grid()
        frames = run.frames()
        units = run.fields[frames[0]].properties.velocity_units
        shape = (X.shape[0],X.shape[1],len(frames))
        U = np.zeros(shape)
        V = np.zeros(shape)
        for ind in range(len(frames)):                
            x,y,u,v = run.fields[frames[ind]].create_mesh_grid()
            U[:,:,ind] = u[::-1]
            V[:,:,ind] = v[::-1]
    
    else:
        print('Not all run in same grid')
        return
    
    Q = ax.quiver(X,Y,U[...,0],V[...,0],np.sqrt(U[...,0]**2+V[...,0]**2),angles='xy',cmap='coolwarm')
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'1 '+units, labelpos='E',coordinates='figure')
    
    def update_quiver(i,Q,U,V):
        Q.set_UVC(U[...,i],V[...,i],np.sqrt(U[...,i]**2+V[...,i]**2))
        ax.set_title('Velocity field frame number: %d' % i ) 
        return Q,
    
    anim = animation.FuncAnimation(fig, update_quiver,frames=shape[2],interval=300,fargs=(Q,U,V), blit=False,repeat=True)
    fig.tight_layout()
    plt.show()
    return anim

def mean_run_profile(run,axis='y'):
    U_mean_profile,U_number_of_vectors,V_mean_profile,V_number_of_vectors,grid_points_agp = run.mean_profile(axis)
    v_units = run.fields[run.frames()[0]].properties.velocity_units
    l_units = run.fields[run.frames()[0]].properties.length_unit
    plot(U_mean_profile , grid_points_agp,'Mean run plot in the '+axis+' direction',[r'$\bar{u}$ '+v_units,l_units])


def run_vorticity_animation(run):
    fig, ax = plt.subplots(figsize=(21, 18))
    if run.check_same_grid_run():
        X,Y = run.run_grid()
        X=X[2:-2,2:-2]
        Y=Y[2:-2,2:-2]
        frames = run.frames()
        shape = (X.shape[0],X.shape[1],len(frames))
        W = np.zeros(shape)
        for ind in range(len(frames)):                
            w_c = run.fields[frames[ind]].vorticity_field()
            W[:,:,ind] = w_c[:,:]
    
    else:
        print('Not all run in same grid')
        return
    
    def update_cont(i):
        ax.clear()
        ax.contourf(X,Y,W[:,:,i])
        ax.set_title('Vorticity field frame number: %d' % i ) 
    
    anim = animation.FuncAnimation(fig, update_cont,interval=400, blit=False,repeat=True)
    fig.tight_layout()
    plt.show()
    return anim,W
 
def mean_run_vorticity(run,square=1):
    X,Y,W = run.run_vorticity_field()
    contour(X,Y,W**square,'Mean run vorticity field')

def mean_reynolds_stress(run,direction='xy'):
    X,Y = run.run_grid()
    r_stress = run.run_reynolds_stress()
    contour(X,Y,r_stress,'Mean run reynolds stress in the '+direction+' direction')

def mean_run_quiver(run):
    X,Y = run.run_grid()
    U,V = run.run_mean_velocities()
    U[U>2.5]=np.nan
    
    title = 'Mean velocity (U_wind = 2.5 m/s)'
    units = run.fields[run.frames()[0]].properties.velocity_units
    quiver(X,Y,U,V,title,units)
    
def mean_run_quiver_profile(run,loc_per,axis='y'):
    X,Y = run.run_grid()
    U,V = run.run_mean_velocities()
    if axis=='x':
        border = int(X.shape[0]*(loc_per/100))
        grid_points= X[border,:]
        V_prof = np.nanmean(V[border-1:border+2,:],axis=0)
        title = 'Mean velocity porfile in the '+axis+' direction over all frames at '+str(loc_per)+'% of area'
        units = run.fields[run.frames()[0]].properties.velocity_units
        quiver_profile(grid_points,V_prof,title,units)
    else:
        border = int(Y.shape[1]*(loc_per/100))
        grid_points= Y[:,border]
        U_prof = np.nanmean(U[:,border-1:border+2],axis=1)
        title = 'Mean velocity porfile in the '+axis+' direction over all frames at '+str(loc_per)+'% of area'
        units = run.fields[run.frames()[0]].properties.velocity_units
        quiver_profile(grid_points,U_prof,title,units)

def mean_vorticity_profile(run,loc_per,axis='y'):
    X,Y,W = run.run_vorticity_field()
    if axis=='x':
        border = int(X.shape[0]*(loc_per/100))
        grid_points= X[border,:]
        W_prof = np.nanmean(W[border-1:border+2,:],axis=0)
        title = 'Mean vorticity porfile in the '+axis+' direction over all frames at '+str(loc_per)+'% of area'
        l_units = run.fields[run.frames()[0]].properties.length_unit
        plot(grid_points,W_prof,title,(l_units,'vorticity'))
    else:
        border = int(Y.shape[1]*(loc_per/100))
        grid_points= Y[:,border]
        W_prof = np.nanmean(W[:,border-1:border+2],axis=1)
        title = 'Mean vorticity porfile in the '+axis+' direction over all frames at '+str(loc_per)+'% of area'
        l_units = run.fields[run.frames()[0]].properties.length_unit
        plot(W_prof,grid_points,title,('vorticity',l_units))


def mean_reynolds_stress_profile(run,loc_per,axis='y'):
    X,Y = run.run_grid()
    r_stress = run.run_reynolds_stress()
    if axis=='x':
        border = int(X.shape[0]*(loc_per/100))
        grid_points= X[border,:]
        r_stress_prof = np.nanmean(r_stress[border-1:border+2,:],axis=0)
        title = 'Mean Reynolds stress porfile in the '+axis+' direction over all frames at '+str(loc_per)+'% of area'
        l_units = run.fields[run.frames()[0]].properties.length_unit
        plot(grid_points,r_stress_prof,title,(l_units,'Reynold Stress'))
    else:
        border = int(Y.shape[1]*(loc_per/100))
        grid_points= Y[:,border]
        r_stress_prof = np.nanmean(r_stress[:,border-1:border+2],axis=1)
        title = 'Mean Reynolds stress porfile in the '+axis+' direction over all frames at '+str(loc_per)+'% of area'
        l_units = run.fields[run.frames()[0]].properties.length_unit
        plot(r_stress_prof,grid_points,title,('Reynold Stress',l_units))
