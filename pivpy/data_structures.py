
import numpy as np
import copy
import scipy
from scipy.stats import norm
from scipy import io,signal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from weighted_median import *

def check_str_bool(s):
    return s in ['True' ,'true', '1', 't', 'y','YES' ,'Yes','yes', 'yeah','Yeah', 'yup', 'certainly', 'uh-huh']

class vec_properties:
    def __init__(self,source,ws,time_unit,time_scale_to_seconds,length_unit,length_scale_to_meter):
        self.source = source
        self.ws = ws
        self.time_unit = time_unit
        self.time_scale_to_seconds = time_scale_to_seconds
        self.length_unit = length_unit
        self.length_scale_to_meter = length_scale_to_meter
        self.velocity_units = length_unit+'/'+time_unit
        
        
    def show(self):
        print(
        'source: ',self.source,'\n',
        'window size: ',self.ws,'\n',
        'dt: ',self.time_scale_to_seconds,'\n',
        'pixel to meter: ',self.length_scale_to_meter,'\n',
        'velocity units: ',self.velocity_units,'\n')


class field_properties:
    def __init__(self,frame,time,images_path,source,time_unit,time_scale_to_seconds,length_unit,length_scale_to_meter):
        self.frame = frame
        self.time = time
        self.images_path = images_path
        self.source = source
        self.history = ''
        self.time_unit = time_unit
        self.time_scale_to_seconds = time_scale_to_seconds
        self.length_unit = length_unit
        self.length_scale_to_meter = length_scale_to_meter
        self.velocity_units = length_unit+'/'+time_unit
    
    def show(self):
        print(
        'frame: ',self.frame,'\n',
        'absolute time: ',self.time,'\n',
        'images_path: ',self.images_path,'\n',
        'source: ',self.source,'\n',
        'dt: ',self.time_scale_to_seconds,'\n',
        'pixel to meter: ',self.length_scale_to_meter,'\n',
        'length units: ',self.length_scale_to_meter,'\n',
        'velocity units: ',self.velocity_units)

class run_properties:
    pass
        
class vector:
    def __init__(self,X,Y,U,V,S2N,properties):
        self.X = X
        self.Y = Y
        self.U = U
        self.V = V
        self.S2N = S2N
        self.properties = properties
    
    def convert_units(self,output_length_unit,output_time_unit):
        LS = {'mm':0.001, 'cm':0.01, 'm':1.0,'meter':1.0,'meters':1.0, 'km':1000.}
        TS = {'ms':0.001, 's':1.0,'second':1.0,'seconds':1.0, 'min':60.,'mins':60.,'h':3600.,'hour':3600.,'hours':3600.}
        LS[self.properties.length_unit]=float(self.properties.length_scale_to_meter)
        TS[self.properties.time_unit]=float(self.properties.time_scale_to_seconds)
        
        self.X = self.X*(LS[self.properties.length_unit]/LS[output_length_unit])
        self.Y = self.Y*(LS[self.properties.length_unit]/LS[output_length_unit])

        self.U = self.U*(LS[self.properties.length_unit]/LS[output_length_unit])*(TS[output_time_unit]/TS[self.properties.time_unit])
        self.V = self.V*(LS[self.properties.length_unit]/LS[output_length_unit])*(TS[output_time_unit]/TS[self.properties.time_unit])
        
        self.properties.length_unit = output_length_unit
        self.properties.length_scale_to_meter = LS[output_length_unit]
        self.properties.time_unit = output_time_unit
        self.properties.time_scale_to_seconds = TS[output_time_unit]
        self.properties.velocity_units = output_length_unit+'/'+output_time_unit
    
    
class field:
    def __init__(self,field_properties):  
        self.data = {}
        self.filtered = {}
        self.properties = field_properties


    def __add__(self,other):
        check_list = []
        check_list.append(self.properties.length_unit == other.properties.length_unit)
        check_list.append(self.properties.length_scale_to_meter == other.properties.length_scale_to_meter)
        check_list.append(self.properties.time_unit == other.properties.time_unit)
        check_list.append(self.properties.time_scale_to_seconds == other.properties.time_scale_to_seconds)
        check_list.append(self.properties.velocity_units == other.properties.velocity_units)

        if all(check_list):
            sum_properties = self.properties
            sum_properties.source = 'Sum'
            sum_properties.frame = self.properties.frame + ' & ' + other.properties.frame
            sum_properties.time = self.properties.time + ' & ' + other.properties.time
            sum_properties.images_path = self.properties.images_path + ' & ' + other.properties.images_path

            sum_field = field(sum_properties)
            for xy in list(self.data.keys()):
                sum_field.add_vec(self.data[xy])
            for xy in list(other.data.keys()):
                sum_field.add_vec(other.data[xy])
    
            return sum_field

        else:
            print( 'Field properties do not match')

    def add_vec(self, vector):
        self.data[vector.X,vector.Y] = vector

    def check_if_grid_point_exists(self,x,y):
        xy = list(self.data.keys())
        return (x,y) in xy
        
    def move_to_filtered(self,vector):
        self.filtered[vector.X,vector.Y] = copy.deepcopy(vector)
        vector.U = np.nan
        vector.V = np.nan
        vector.properties.source = 'filtered'

    def transfer(self,other):
        for xy in list(other.data.keys()):
            self.add_vec(other.data[xy])
    
    def convert_field_units(self,output_length_unit,output_time_unit):
        XY = list(self.data.keys())
        
        if self.properties.length_unit == None or self.properties.length_unit == '':
            self.properties.length_unit = str(input('field length units'))    
        if self.properties.length_scale_to_meter== None or self.properties.length_scale_to_meter == '':
            self.length_scale_to_meter = str(input('field length units to meters'))
        if self.properties.time_unit == None or self.properties.time_unit == '':
            self.properties.time_unit = str(input('field time units'))
        if self.properties.time_scale_to_seconds== None or self.properties.time_scale_to_seconds == '':
            self.properties.time_scale_to_seconds = str(input('field time units to seconds'))
        
        for xy in XY:
            self.data[xy].properties.length_unit = self.properties.length_unit
            self.data[xy].properties.length_scale_to_meter = self.properties.length_scale_to_meter
            self.data[xy].properties.time_unit = self.properties.time_unit
            self.data[xy].properties.time_scale_to_seconds = self.properties.time_scale_to_seconds
            self.data[xy].convert_units(output_length_unit,output_time_unit)
            self.add_vec(self.data[xy])
            self.remove_vec(xy[0],xy[1])
        
        XY0 = list(self.data.keys())[0]
        self.properties.length_unit = self.data[XY0].properties.length_unit
        self.properties.length_scale_to_meter = self.data[XY0].properties.length_scale_to_meter
        self.properties.time_unit = self.data[XY0].properties.time_unit
        self.properties.time_scale_to_seconds = self.data[XY0].properties.time_scale_to_seconds
        self.properties.velocity_units = self.data[XY0].properties.velocity_units
        
    def remove_vec(self,X,Y,vector=None):
        if vector is not None:
            del self.data[vector.X,vector.Y]
        else:
            del self.data[X,Y]
    
    def return_vel(self,x,y):
        u = self.data[x,y].U
        v = self.data[x,y].V

        return u,v

    def return_n_closest_neighbors(self,x,y,n=4):
        X,Y = self.return_grid()
        dist = np.sqrt((X-x)**2+(Y-y)**2)
        n_closest_neighbors = [ [(X[ind],Y[ind]),dist[ind]] for ind in dist.argsort()[:n]]
        
        return n_closest_neighbors

    def return_closest_neighbors_radius(self,x,y,radius):
        X,Y = self.return_grid()
        dist = np.sqrt((X-x)**2+(Y-y)**2)
        indecies = np.where(dist<radius)
        closest_neighbors = [[(X[indecies[0][i]],Y[indecies[0][i]]),dist[indecies[0][i]]] for i in range(len(indecies[0]))]
        return closest_neighbors

    def return_grid(self):
        XY = list(self.data.keys())
        X,Y = zip(*XY)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y
    
    def return_all_velocities(self):
        XY = list(self.data.keys())
        U = np.array([self.data[xy[0],xy[1]].U for xy in XY])
        V = np.array([self.data[xy[0],xy[1]].V for xy in XY])
        return U,V

    def sub_average(self):
        XY = list(self.data.keys())
        umean,ustd,vmean,vstd = self.mean_velocity()
        for i in range(len(XY)):
            self.data[XY[i]].U = self.data[XY[i]].U - umean
            self.data[XY[i]].V = self.data[XY[i]].V - vmean

    def create_mesh_grid(self):
        X,Y = self.return_grid()
        U,V = self.return_all_velocities()
        X_mesh_grid = sorted(list(set(X)))
        Y_mesh_grid = sorted(list(set(Y)))
        X_mesh_grid,Y_mesh_grid = np.meshgrid(X_mesh_grid,Y_mesh_grid)
        U_mesh_grid = np.empty(X_mesh_grid.shape)
        U_mesh_grid.fill(np.nan)
        V_mesh_grid = np.empty(X_mesh_grid.shape)
        V_mesh_grid.fill(np.nan)
        for vec_ind in range(len(X)):
            x = X[vec_ind]
            y = Y[vec_ind]
            col = np.array(np.where(X_mesh_grid[0,:]==x))[0,0]
            row = np.array(np.where(Y_mesh_grid[:,0]==y))[0,0]
            U_mesh_grid[row,col] = U[vec_ind]
            V_mesh_grid[row,col] = V[vec_ind]
        
        return X_mesh_grid,Y_mesh_grid[::-1],U_mesh_grid[::-1],V_mesh_grid[::-1]

    def s2n_filter(self,threshold):
        XY = list(self.data.keys())
        for xy in XY:
            if self.data[xy].S2N < threshold:
                self.move_to_filtered(self.data[xy])
        
    def hist_filter(self,percentage):
        def TrueXor(*args):
            return sum(args) == 1
        
        hist_u,hist_v,hist2d = self.velocity_histogram()
        #strech boundry edges
        hist_u[1][0] = hist_u[1][0]-1
        hist_u[1][-1] = hist_u[1][-1]+1
        hist_v[1][0] = hist_v[1][0]-1
        hist_v[1][-1] = hist_v[1][-1]+1
        
        hist2d[1][0] = hist2d[1][0]-1
        hist2d[1][-1] = hist2d[1][-1]+1
        hist2d[2][0] = hist2d[2][0]-1
        hist2d[2][-1] = hist2d[2][-1]+1
        
        XY = list(self.data.keys())
        number_of_vectors = len(XY)
        for xy in XY:
            u = self.data[xy].U
            v = self.data[xy].V
            if np.isfinite(u) and not np.isfinite(v):
                if hist_u[0][np.digitize(u,hist_u[1])-1] / number_of_vectors > percentage/100:
                    u_iter,v_iter = self.inverse_distance_interpolation(xy[0],xy[1])
                    if np.isfinite(v_iter):
                        self.data[xy].V = v_iter
                        v = v_iter
                    else:
                        self.move_to_filtered(self.data[xy])
            
            if np.isfinite(v) and not np.isfinite(u):
                if hist_v[0][np.digitize(v,hist_v[1])-1] / number_of_vectors > percentage/100:
                    u_iter,v_iter = self.inverse_distance_interpolation(xy[0],xy[1])
                    if np.isfinite(u_iter):
                        self.data[xy].U = u_iter
                        u = u_iter
                    else:
                        self.move_to_filtered(self.data[xy])
            if np.isfinite(v) and np.isfinite(u):
                U_histpos = np.digitize(u,hist2d[1])-1
                V_histpos = np.digitize(v,hist2d[2])-1
                if hist2d[0][U_histpos,V_histpos] / number_of_vectors < percentage/100:
                    self.move_to_filtered(self.data[xy])
    
    def Z_filter(self,threshold,neighbors=4,power=1):
        XY = list(self.data.keys())
        for xy in XY:
            u = self.data[xy].U
            v = self.data[xy].V
            closest_neighbors = self.return_n_closest_neighbors(self.data[xy].X,self.data[xy].Y,neighbors+1)[1:]
            neighbor_pos , dis = zip(*closest_neighbors)
            weights = [(1/d)**power for d in dis]
            U,V = zip(*[self.return_vel(pos[0],pos[1]) for pos in neighbor_pos])

            median_U = weighted_median(U,weights)
            median_V = weighted_median(V,weights)
            median_absolute_deviation_U = weighted_median([np.abs(u_neighbor - median_U) for u_neighbor in U],weights)
            median_absolute_deviation_V = weighted_median([np.abs(v_neighbor - median_V) for v_neighbor in V],weights)

            if 0.6745*(u - median_U) / max(median_absolute_deviation_U,0.01) > threshold:
                self.move_to_filtered(self.data[xy])
                continue

            if 0.6745*(v - median_V) / max(median_absolute_deviation_V,0.01) > threshold:
                self.move_to_filtered(self.data[xy])
                continue
    
    def max_arg_filter(self,U_bound,V_bound):
        XY = list(self.data.keys())
        for xy in XY:
            U_check = True
            V_check = True
            if self.data[xy].U > U_bound[1] or self.data[xy].U < U_bound[0]:
                U_check=False
            if self.data[xy].V > V_bound[1] or self.data[xy].V < V_bound[0]:
                V_check=False
            
            if U_check and not V_check:
                u_itr,v_itr = self.inverse_distance_interpolation(xy[0],xy[1])
                self.data[xy].V = v_itr
            elif V_check and not U_check:
                u_itr,v_itr = self.inverse_distance_interpolation(xy[0],xy[1])
                self.data[xy].U = u_itr
            elif not V_check and not U_check:
                self.move_to_filtered(self.data[xy])
            
    def mean_velocity(self):
        U,V = self.return_all_velocities()
        return np.nanmean(U),np.nanstd(U),np.nanmean(V),np.nanstd(V)
    
    def velocity_histogram(self,bins=10):
        def remove_nans(u,v):
            u = list(u)
            v = list(v)
            nan_index=[]
            for i in range(len(u)):
                if not np.isfinite(u[i]) or not np.isfinite(v[i]):
                    nan_index.append(i)
            for index in sorted(nan_index, reverse=True):
                del u[index]
                del v[index]
            return np.array(u),np.array(v)
        U,V = self.return_all_velocities()
        hist_U = np.histogram(U[np.isfinite(U)],bins)
        hist_V = np.histogram(V[np.isfinite(V)],bins)
        U,V = remove_nans(U,V)
        hist2d = np.histogram2d(U, V, bins)
        return hist_U,hist_V,hist2d
    
    def extract_area(self,x_boundry,y_boundry):
        area = field(self.properties)
        X,Y = self.return_grid()
        for i in range(len(X)):
            if x_boundry[0]<=X[i]<=x_boundry[1] and y_boundry[0]<=Y[i]<=y_boundry[1]:
                area.add_vec(self.data[X[i],Y[i]])

        return area

    def vel_gradients(self):
        X,Y,U,V = self.create_mesh_grid()
        Udx,Udy = np.gradient(U)
        Vdx,Vdy = np.gradient(V)
        
        return X,Y,Udx,Udy,Vdx,Vdy
    
    def vel_differntial(self):
        def least_square_diff(field,grid,axis=0):
            if axis==0:
                shape = field.shape
                dif = np.zeros(shape)
                for row in range(shape[0]):
                    for col in range(2,shape[1]-2):
                        rs = 2*field[row,col+2]+field[row,col+1]
                        ls = -field[row,col-1]-2*field[row,col-2]
                        dis = 10*(grid[row,col+1]-grid[row,col])
                        dif[row,col] = (rs+ls)/dis
                        #dif[row,col] = (2*field[row,col+2]+field[row,col+1]-field[row,col-1]-2*field[row,col-2])/10*(grid[row,col+1]-grid[row,col])
                return dif
                
            elif axis==1:
                shape = field.shape
                dif = np.zeros(shape)
                for row in range(2,shape[0]-2):
                    for col in range(shape[1]):
                        us = 2*field[row-2,col]+field[row-1,col]
                        ds = -field[row+1,col]-2*field[row+2,col]
                        dis = 10*(grid[row-1,col]-grid[row,col])
                        dif[row,col] = (us+ds)/dis
                        #dif[row,col] = (2*field[row-2,col]+field[row-1,col]-field[row+1,col]-2*field[row+2,col])/10*(grid[row-1,col]-grid[row,col])
                return dif
                
        X,Y,U,V = self.create_mesh_grid()
        dU_x = least_square_diff(U,X)
        dU_y = least_square_diff(U,Y,axis=1)
        dV_x = least_square_diff(V,X)
        dV_y = least_square_diff(V,Y,axis=1)
        
        return dU_x,dU_y,dV_x,dV_y

    def profile(self,axis='y'):
        X,Y,U,V = self.create_mesh_grid()
        if axis=='y' or axis=='Y':
            U_profile = np.nanmean(U,axis=1)[::-1]
            V_profile = np.nanmean(V,axis=1)[::-1]
            Y_profile = Y[:,0]

            return U_profile,V_profile,Y_profile
        else: 
            U_profile = np.nanmean(U,axis=0)[::-1]
            V_profile = np.nanmean(V,axis=0)[::-1]
            X_profile = X[0,:]

            return U_profile,V_profile,X_profile
    
    def vorticity_field(self):
        dU_x,dU_y,dV_x,dV_y = self.vel_differntial()
        vort = dV_x-dU_y
        return vort[2:-2,2:-2]
        
    def inverse_distance_interpolation(self,x,y,number_of_neighbors=5,radius=None,inverse_power=2):
        def weigted_velocity(neighbors_vels,weights):
            weight_sum=0
            weigted_vels=[]
            for i in range(len(neighbors_vels)):
                if not np.isnan(neighbors_vels[i]):
                    weight_sum += weights[i]
                    weigted_vels.append(weights[i]*neighbors_vels[i])
            return np.nansum(weigted_vels)/weight_sum

        if self.check_if_grid_point_exists(x,y):
            if radius is not None:
                indecies,distances = zip(*self.return_closest_neighbors_radius(x,y,radius))
            else:
                indecies,distances = zip(*self.return_n_closest_neighbors(x,y,n=number_of_neighbors+1))
            weights = list(np.array(distances[1:])**-float(inverse_power))
            neighbors_vel = [self.return_vel(ind[0],ind[1]) for ind in indecies[1:]]

            u_vels,v_vels = zip(*neighbors_vel)

            inter_u = weigted_velocity(u_vels,weights)
            inter_v = weigted_velocity(v_vels,weights)

            return inter_u,inter_v
        else:
            if radius is not None:
                indecies,distances = zip(*self.return_closest_neighbors_radius(x,y,radius))
            else:
                indecies,distances = zip(*self.return_n_closest_neighbors(x,y,n=number_of_neighbors))
            weights = list(np.array(distances)**-float(inverse_power))
            neighbors_vel = [self.return_vel(ind[0],ind[1]) for ind in indecies]
            
            u_vels,v_vels = zip(*neighbors_vel)

            inter_u = weigted_velocity(u_vels,weights)
            inter_v = weigted_velocity(v_vels,weights)

            return inter_u,inter_v
    
    def interpf(self):
        X,Y = self.return_grid()
        for ind in range(X.shape[0]):
            pos = (X[ind],Y[ind])
            u_cur,v_cur = self.return_vel(pos[0],pos[1])
            if np.isnan(u_cur) and np.isnan(v_cur):
                u_iter,v_iter = self.inverse_distance_interpolation(pos[0],pos[1])
                vec = self.data[pos]
                vec.U =  u_iter
                vec.V =  v_iter
                vec.properties.source = 'Interpolation'
            elif np.isnan(u_cur):
                u_iter,v_iter = self.inverse_distance_interpolation(pos[0],pos[1])
                vec = self.data[pos]
                vec.U =  u_iter
                vec.properties.source = 'Interpolation'
            elif np.isnan(v_cur):
                u_iter,v_iter = self.inverse_distance_interpolation(pos[0],pos[1])
                vec = self.data[pos]
                vec.V =  v_iter
                vec.properties.source = 'Interpolation'
            
    def remap(self,X,Y,shape_of_new_grid=None):
        new_feild = field(self.properties)
        Xold,Yold = self.return_grid()
        if shape_of_new_grid==None:
            X = X.flatten()
            Y = Y.flatten()
        else:
            X,Y = np.meshgrid(np.linspace(Xold.min(),Xold.max(),shape_of_new_grid[1]),np.linspace(Yold.min(),Yold.max(),shape_of_new_grid[0]))
            X = X.flatten()
            Y = Y.flatten()
        vec_properties = self.data[Xold[0],Yold[0]].properties
        vec_properties.source = 'Interpolation'
        for ind in range(len(X)):
            u,v = self.inverse_distance_interpolation(X[ind],Y[ind])
            vec = vector(X[ind],Y[ind],u,v,0,vec_properties)
            new_feild.add_vec(vec)
        
        self.filtered = self.data
        self.data = {}
        self.transfer(new_feild)

    def auto_spatial_correlation(self):
        X,Y,U,V = self.create_mesh_grid()
        Uc = scipy.signal.convolve2d(U,U[::-1])
        Vc = scipy.signal.convolve2d(V,V[::-1])
        Uc = Uc - Uc.min()
        Vc = Vc - Vc.min()
        s_cor = np.sqrt(Uc**2+Vc**2)
        dX = X - np.mean(X[0,:])
        dY = Y - np.mean(Y[:,0])
        return dX,dY,s_cor


class run:
    def __init__(self):  
        self.fields = {}
    
    def add_field(self,field):
        self.fields[field.properties.frame] = field
    
    def frames(self):
        return list(self.fields.keys())
        
    def remove_field(self,frame,field=None):
        if field is not None:
            del self.fields[field.properties.frame]
        else:
            del self.fields[frame]

    def remap_run(self,X,Y,shape_of_new_grid=None):
        frames = self.frames()
        for frame in frames:
            self.fields[frame].remap(X,Y,shape_of_new_grid)
            
    def convert_run_units(self,output_length_unit,output_time_unit,run_length_unit=None,run_length_scale_to_meter=None,run_time_unit=None,run_time_scale_to_seconds=None):
        same_prop = check_str_bool(input('Do all frames in run have the same properties?'))
        if same_prop:
            ''' After correcting the properties of run use this:
            if self.properties.length_unit == None or self.properties.length_unit == '':
                self.properties.length_unit = str(input('run length units: '))    
            if self.properties.length_scale_to_meter== None or self.properties.length_scale_to_meter == '':
                self.properties.length_scale_to_meter = str(input('run length units to meters: '))
            if self.properties.time_unit == None or self.properties.time_unit == '':
                self.properties.time_unit = str(input('run time units: '))
            if self.properties.time_scale_to_seconds== None or self.properties.time_scale_to_seconds == '':
                self.properties.time_scale_to_seconds = str(input('run time units to seconds: '))
            '''
            if run_length_unit is None:
                self.length_unit = str(input('run length units: '))
            else:
                self.length_unit = run_length_unit
                
            if run_length_scale_to_meter is None:
                self.length_scale_to_meter = str(input('run length units to meters: '))
            else:
                self.length_scale_to_meter = run_length_scale_to_meter
                
            if run_time_unit is None:
                self.time_unit = str(input('run time units: '))
            else:
                self.time_unit = run_time_unit
                
            if run_time_scale_to_seconds is None:
                self.time_scale_to_seconds = str(input('run time units to seconds: '))
            else:
                self.time_scale_to_seconds = run_time_scale_to_seconds
        
        frames = self.frames()
        for frame in frames:
            if same_prop:
                self.fields[frame].properties.length_unit = self.length_unit
                self.fields[frame].properties.length_scale_to_meter = self.length_scale_to_meter
                self.fields[frame].properties.time_unit = self.time_unit
                self.fields[frame].properties.time_scale_to_seconds = self.time_scale_to_seconds
            self.fields[frame].convert_field_units(output_length_unit,output_time_unit)
            
    def check_same_grid_run(self):
        frames = self.frames()
        base_frame = frames[0]
        for frame in frames:
            X_base,Y_base = self.fields[base_frame].return_grid()
            X_check,Y_check = self.fields[frame].return_grid()
            if all(X_base == X_check) and all(Y_base == Y_check):
                base_frame = frame
            else:
                return False
        return True

    def gp_exists_all_frames(self,x,y,show_missing_frames=False):
        frames = self.frames()
        gp_exists = [self.fields[f].check_if_grid_point_exists(x,y) for f in frames]
        if all(gp_exists):
            return True
        else:
            no_gp_frames = [x for x, y in zip(frames, gp_exists) if y == False]
            frames_with_gp = [x for x, y in zip(frames, gp_exists) if y == True]
            #allows checking of misssing grid point frames
            if show_missing_frames:
                print('Frames without the requested grid point ','(',x,',',y,')',': ',no_gp_frames)
            return frames_with_gp

    def run_grid(self):
        frames = self.frames()
        Y_agp = []
        X_agp =[]
        for frame in frames:
            X,Y = self.fields[frame].return_grid()
            Y_agp += Y.tolist()
            Y_agp = sorted(list(set(Y_agp)))
            X_agp += X.tolist()
            X_agp = sorted(list(set(X_agp)))
        
        return np.meshgrid(np.array(X_agp),np.array(Y_agp))

    def grid_point_velocity(self,x,y,frames=None):
        if frames==None:
            frames = self.frames()    
            if self.gp_exists_all_frames(x,y):
                U = []
                V = []
                for f in frames:
                    u,v = self.fields[f].return_vel(x,y)
                    U.append(u)
                    V.append(v)
                    
                U = np.array(U)
                V = np.array(V)
                return U,V
        else:
            U = []
            V = []
            for f in frames:
                u,v = self.fields[f].return_vel(x,y)
                U.append(u)
                V.append(v)
                
            U = np.array(U)
            V = np.array(V)
            return U,V
    
    def return_field(self,number_of_field,name_of_frame=None):
        if name_of_frame is not None:
            return self.fields[name_of_frame]
        else:
            return self.fields[self.frames()[number_of_field]]

    def mean_gp_velocity(self,x,y):
        for_all_frames = self.gp_exists_all_frames(x,y)
        if for_all_frames==True:
            U,V = self.grid_point_velocity(x,y)
            U_rms = U - np.nanmean(U)
            V_rms = V - np.nanmean(V)
            return np.nanmean(U),U_rms,np.nanmean(V),V_rms
        else:
            U,V = self.grid_point_velocity(x,y,for_all_frames)
            U_rms = U - np.nanmean(U)
            V_rms = V - np.nanmean(V)
            return np.nanmean(U),U_rms,np.nanmean(V),V_rms

    def mean_velocity_properties(self):
        frames = self.frames()
        U_mean = []
        V_mean = []
        for f in frames:
            u_mean,u_std,v_mean,v_std = self.fields[f].mean_velocity()
            U_mean.append(u_mean)
            V_mean.append(v_mean)
        Um = np.mean(U_mean)
        Vm = np.mean(V_mean)
        U_rms = [(np.sqrt((u-Um)**2)) for u in U_mean]
        V_rms = [(np.sqrt((v-Vm)**2)) for v in V_mean]
        print('Max in mean U velocity accures in frame: ',frames[U_mean.index(max(U_mean))])
        print('Max in mean V velocity accures in frame: ',frames[V_mean.index(max(V_mean))])
        U_mean = np.array(U_mean)
        V_mean = np.array(V_mean)
        U_rms = np.array(U_rms)
        V_rms = np.array(V_rms)
        return U_mean,U_rms,V_mean,V_rms

    def run_mean_velocities(self):
        if self.check_same_grid_run():
            X,Y = self.run_grid()
            frames = self.frames()
            shape = (X.shape[0],X.shape[1],len(frames))
            U_mean = np.zeros(shape)
            V_mean = np.zeros(shape)
            for ind in range(len(frames)):                
                x,y,u,v = self.fields[frames[ind]].create_mesh_grid()
                U_mean[:,:,ind] = u[::-1]
                V_mean[:,:,ind] = v[::-1]
            return np.nanmean(U_mean,axis=2),np.nanmean(V_mean,axis=2)
        else:
            X,Y = self.run_grid()
            U_mean = np.zeros(X.shape)
            V_mean = np.zeros(Y.shape)
            for row in range(X.shape[0]):
                for col in range(X.shape[1]):
                    u,urms,v,vrms = self.mean_gp_velocity(X[row,col],Y[row,col])
                    U_mean[row,col] = u
                    V_mean[row,col] = v

            return U_mean,V_mean
    
    def run_reynolds_stress(self,direction='xy'):
        X,Y = self.run_grid()
        rstress = np.zeros(X.shape)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                #check equation - do you need to multiply by density??
                u,urms,v,vrms = self.mean_gp_velocity(X[row,col],Y[row,col])
                if direction=='xy' or direction=='yx':
                    rstress[row,col] = np.nanmean(np.multiply(urms,vrms))
                elif direction=='xx':
                    rstress[row,col] = np.nanmean(np.multiply(urms,urms))
                elif direction=='yy':
                    rstress[row,col] = np.nanmean(np.multiply(vrms,vrms))
        return rstress

    def frame_reynolds_stress(self,frame,direction='xy'):
        X,Y,U,V = self.fields[frame].create_mesh_grid()
        rstress = np.zeros(X.shape)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                #check equation - do you need to multiply by density??
                u,urms,v,vrms = self.mean_gp_velocity(X[row,col],Y[row,col])
                if direction=='xy' or direction=='yx':
                    rstress[row,col] = (U[row,col] - u)*(V[row,col] - v)
                elif direction=='xx':
                    rstress[row,col] = (U[row,col] - u)**2
                elif direction=='yy':
                    rstress[row,col] = (V[row,col] - v)**2
        return rstress

        
    def mean_profile(self,axis='y'):
        frames = self.frames()
        if axis=='y' or axis=='Y':
            Y_agp = []
            for frame in frames:
                X,Y = self.fields[frame].return_grid()
                Y_agp += Y.tolist()
                Y_agp = sorted(list(set(Y_agp)))
                
            U_profiles = np.empty((len(Y_agp),len(frames)))
            U_profiles.fill(np.nan)
            V_profiles = np.empty((len(Y_agp),len(frames)))
            V_profiles.fill(np.nan)
            Y_agp = np.array(Y_agp)
            
            for col_ind,frame in list(enumerate(frames)):
                U_cur_prof,V_cur_prof,Y_cur_prof = self.fields[frame].profile(axis=axis)
                for i in range(len(Y_cur_prof)):
                     row_ind = np.where(Y_agp==Y_cur_prof[i])[0][0]
                     U_profiles[row_ind,col_ind] = U_cur_prof[i]
                     V_profiles[row_ind,col_ind] = V_cur_prof[i]
                     
            U_mean_profile = np.nanmean(U_profiles,axis=1)[::-1]
            U_number_of_vectors = np.sum(np.invert(np.isnan(U_profiles)),1)
            V_mean_profile = np.nanmean(V_profiles,axis=1)[::-1]
            V_number_of_vectors = np.sum(np.invert(np.isnan(V_profiles)),1)
            
            return U_mean_profile,U_number_of_vectors,V_mean_profile,V_number_of_vectors,Y_agp
        
        else:
            X_agp = []
            for frame in frames:
                X,Y = self.fields[frame].return_grid()
                X_agp += X.tolist()
                X_agp = sorted(list(set(X_agp)))
                
            U_profiles = np.empty((len(X_agp),len(frames)))
            U_profiles.fill(np.nan)
            V_profiles = np.empty((len(X_agp),len(frames)))
            V_profiles.fill(np.nan)
            X_agp = np.array(X_agp)
            
            for col_ind,frame in list(enumerate(frames)):
                U_cur_prof,V_cur_prof,X_cur_prof = self.fields[frame].profile(axis=axis)
                for i in range(len(X_cur_prof)):
                     row_ind = np.where(X_agp==X_cur_prof[i])[0][0]
                     U_profiles[row_ind,col_ind] = U_cur_prof[i]
                     V_profiles[row_ind,col_ind] = V_cur_prof[i]
                     
            U_mean_profile = np.nanmean(U_profiles,axis=1)[::-1]
            U_number_of_vectors = np.sum(np.invert(np.isnan(U_profiles)),1)
            V_mean_profile = np.nanmean(V_profiles,axis=1)[::-1]
            V_number_of_vectors = np.sum(np.invert(np.isnan(V_profiles)),1)
            
            return U_mean_profile,U_number_of_vectors,V_mean_profile,V_number_of_vectors,X_agp
        

    def corr_s(self,x1,y1,x2,y2):
        if self.gp_exists_all_frames(x1,y1) and self.gp_exists_all_frames(x2,y2):
            Umean1,U1,Vmean1,V1 = self.mean_gp_velocity(x1,y1)
            Umean2,U2,Vmean2,V2 = self.mean_gp_velocity(x2,y2)

            return np.inner(U1, U2)/(np.sqrt(np.inner(U1, U1)*np.inner(U2, U2))) , np.inner(V1, V2)/(np.sqrt(np.inner(V1, V1)*np.inner(V2, V2)))

        else:
            frames1 = self.gp_exists_all_frames(x1,y1)
            frames2 = self.gp_exists_all_frames(x2,y2)
            frames_for_both = set(frames1).intersection(frames2)
            U1,V1 = self.grid_point_velocity(x1,y1,frames_for_both)
            U2,V2 = self.grid_point_velocity(x2,y2,frames_for_both)
            Umean1,nouse1,Vmean1,nouse2 = self.mean_gp_velocity(x1,y1)
            Umean2,nouse1,Vmean2,nouse2 = self.mean_gp_velocity(x2,y2)
            U1 = U1 - Umean1
            V1 = V1 - Vmean1
            U2 = U2 - Umean2
            V2 = V2 - Vmean2

            return np.inner(U1, U2)/(np.sqrt(np.inner(U1, U1)*np.inner(U2, U2))) , np.inner(V1, V2)/(np.sqrt(np.inner(V1, V1)*np.inner(V2, V2)))
    
    def corr_t(self,x,y,dframes,tau=None):
        if tau is not None:
            dframes = int(tau//float(self.fields[self.frames()[0]].properties.time_scale_to_seconds))
        if self.gp_exists_all_frames(x,y):
            Umean1,U1,Vmean1,V1 = self.mean_gp_velocity(x,y)
            U2 = U1[dframes:]
            V2 = V1[dframes:]
            U1 = U1[:-dframes]
            V1 = V1[:-dframes]
            return np.inner(U1, U2)/(np.sqrt(np.inner(U1, U1)*np.inner(U2, U2))) , np.inner(V1, V2)/(np.sqrt(np.inner(V1, V1)*np.inner(V2, V2)))

        else:
            frames = self.gp_exists_all_frames(x,y)
            U1,V1 = self.grid_point_velocity(x,y,frames)
            U1 = U1 - np.nanmean(U1)
            V1 = V1 - np.nanmean(V1)
            U2 = U1[dframes:]
            V2 = V1[dframes:]
            U1 = U1[:-dframes]
            V1 = V1[:-dframes]
            return np.inner(U1, U2)/(np.sqrt(np.inner(U1, U1)*np.inner(U2, U2))) , np.inner(V1, V2)/(np.sqrt(np.inner(V1, V1)*np.inner(V2, V2)))

    def spatial_corr(self,U1,V1,U2,V2):
        U1 = np.nan_to_num(U1)
        V1 = np.nan_to_num(V1)
        U2 = np.nan_to_num(U2)
        V2 = np.nan_to_num(V2)
        Uc = scipy.signal.convolve2d(U1,U2[::-1])
        Vc = scipy.signal.convolve2d(V1,V2[::-1])
        Uc = Uc - Uc.min()
        Vc = Vc - Vc.min()
        s_cor = np.sqrt(Uc**2+Vc**2)
        return s_cor

    def auto_spatial_correlation(self,U,V):
        U_mean,V_mean = self.run_mean_velocities()
        U_fluc = U-U_mean
        V_fluc = V-V_mean
        return self.spatial_corr(U_fluc,V_fluc,U_fluc,V_fluc)

    def time_spatial_corr(self,starting_frame=0):
        if self.check_same_grid_run():
            frames = self.frames()[starting_frame:]
            U_mean,V_mean = self.run_mean_velocities()
            X,Y,U_base,V_base = self.fields[frames[0]].create_mesh_grid()
            U_base -= U_mean
            V_base -= V_mean
            dx,dy,s_corr = self.fields[frames[0]].auto_spatial_correlation()
            ts_corr = np.zeros((s_corr.shape[0],s_corr.shape[1],len(frames)))
            for ind in range(len(frames)):
                X,Y,U_t,V_t = self.fields[frames[ind]].create_mesh_grid()
                U_t -= U_mean
                V_t -= V_mean
                s_cor = self.spatial_corr(U_base,V_base,U_t,V_t)
                ts_corr[:,:,ind] = s_cor
            
            return dx,dy,ts_corr

        else:
            print('The grid is not constant')
        
    def run_vorticity_field(self):
        if self.check_same_grid_run():
            frames = self.frames()
            X,Y,U,V = self.fields[frames[0]].create_mesh_grid()
            X=X[2:-2,2:-2]
            Y=Y[2:-2,2:-2]
            w0 = self.fields[frames[0]].vorticity_field()
            shape = (w0.shape[0],w0.shape[1],len(frames))
            vort_mean = np.zeros(shape)
            vort_mean[:,:,0] = w0
            for ind in range(1,len(frames)):                
                w = self.fields[frames[ind]].vorticity_field()
                vort_mean[:,:,ind] = w
            return X,Y,np.nanmean(vort_mean,axis=2)
        else:
            print('The grid is not constant')

    
    def tke_frame(self,frame):
        X,Y = self.fields[frame].return_grid()
        U,V = self.fields[frame].return_all_velocities()
        tke = np.zeros(U.shape)
        for ind in range(len(X)):
            umean,urms,vmean,vrms =  self.mean_gp_velocity(X[ind],Y[ind])
            # need to check equation (devide by mean velocities???)
            tke[ind] = 0.5*(np.sqrt((U[ind]-umean)**2+(V[ind]-vmean)**2))

        return X,Y,tke

    def run_tke(self):
        X,Y = self.run_grid()
        tke = np.zeros(X.shape)
        for row in range(X.shape[0]):
            for col in range(X.shape[0]):
                u,urms,v,vrms = self.mean_gp_velocity(X[row,col],Y[row,col])
                # need to check equation (devide by mean velocities???)
                tke[row,col] = np.sqrt((1/len(urms))*np.sum(urms**2+vrms**2))

        return X,Y,tke
    
    def run_filter_outliers(self,s2n_threshold=None,hist_percentage=None,Z_threshold=None,max_U_bound=None,max_V_bound=None):
        frames = self.frames()
        for frame in frames:
            field = self.fields[frame]
            if s2n_threshold is not None:
                field.s2n_filter(s2n_threshold)
            if hist_percentage is not None:
                field.hist_filter(hist_percentage)
            if Z_threshold is not None:
                field.Z_filter(hist_percentage)
            if max_U_bound is not None and max_V_bound is not None:
                field.max_arg_filter(max_U_bound,max_V_bound)
            elif max_U_bound is not None and max_V_bound is None:
                field.max_arg_filter(max_U_bound,[-np.inf,np.inf])
            elif max_U_bound is None and max_V_bound is not None:
                field.max_arg_filter([-np.inf,np.inf],max_V_bound)
        
#debug
'''

frame = 1
dt = 0.001
length_scale = 'pixel'
length_to_meter = 0.00011946

def create_syn_prop(dt,length_scale,length_to_meter):
    global frame
    vprop = vec_properties('Tomers algorithm','64x64','dt',str(dt),str(length_scale),str(length_to_meter))
    cur_frame = (5-len(str(frame)))*str(0)+str(frame)
    fprop = field_properties(cur_frame,frame*dt,'c:/user/user/....','Tomers algorithm','dt',str(dt),str(length_scale),str(length_to_meter))
    frame+=1
    return fprop,vprop


def create_syn_field(rows,cols,xboundries,yboundries,dt,length_scale,length_to_meter,rand_grid_loc=False,number_of_vectors=None):
    fprop,vprop = create_syn_prop(dt,length_scale,length_to_meter)
    field1 = field(fprop)
    if rand_grid_loc==True:
        for i in range(number_of_vectors):
            rand_x_pixel_loc = np.random.randint(xboundries[0],xboundries[1])
            rand_y_pixel_loc = np.random.randint(yboundries[0],yboundries[1])
            U = 5*(np.random.rand()-0.5)
            V = 1*(np.random.rand()-0.5)
            vec = vector(rand_x_pixel_loc,rand_y_pixel_loc,U,V,0,vprop)
            field1.add_vec(vec)
        return field1
    else:
        U = np.zeros((rows,cols))
        V = np.zeros((rows,cols))
        X,Y = np.meshgrid(np.linspace(xboundries[0],xboundries[1],cols),np.linspace(yboundries[0],yboundries[1],rows))
        for row in range(rows):
            for col in range(cols):
                u = 5*(np.random.rand()-0.5)
                v = 1*(np.random.rand()-0.5)
                vec = vector(X[row,col],Y[row,col],u,v,0,vprop)
                field1.add_vec(vec)
        return field1

def create_syn_run(number_of_fields,rows,cols,xboundries,yboundries,dt,length_scale,length_to_meter,rand_grid_loc=False,number_of_vectors=None):
    run1 = run('a')
    for i in range(number_of_fields):
        field1 = create_syn_field(rows,cols,xboundries,yboundries,dt,length_scale,length_to_meter,rand_grid_loc,number_of_vectors)
        run1.add_field(field1)
    return run1


run1 = create_syn_run(200,30,30,[0,1920],[0,1920],dt,length_scale,length_to_meter,rand_grid_loc=False,number_of_vectors=None)

'''
