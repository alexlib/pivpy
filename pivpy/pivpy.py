"""
Created on Sun May 24 22:02:49 2015

@author: Ron, Tomer, Alex
"""
import numpy as np
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter, median_filter
import xarray as xr

""" learn from this example 


import xarray as xr


@xr.register_dataset_accessor('geo')
class GeoAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._center = None

    @property
    def center(self):
        " Return the geographic center point of this dataset."
        if self._center is None:
            # we can use a cache on our accessor objects, because accessors
            # themselves are cached on instances that access them.
            lon = self._obj.latitude
            lat = self._obj.longitude
            self._center = (float(lon.mean()), float(lat.mean()))
        return self._center

    def plot(self):
        " Plot data on a map."
        return 'plotting!'


    In [1]: ds = xr.Dataset({'longitude': np.linspace(0, 10),
   ...:                  'latitude': np.linspace(0, 20)})
   ...: 

In [2]: ds.geo.center
Out[2]: (10.0, 5.0)

In [3]: ds.geo.plot()
Out[3]: 'plotting!'

"""
@xr.register_dataset_accessor('piv')
class PIVAccessor(object):
    def __init__(self,xarray_obj):
        """
        Arguments:
            data : 3D numpy array of rows x cols x 5 matrices:
            [x,y,u,v,chc] 
            x,y,u,v,chc are numpy float 2D arrays of 
            the same size rows x cols  
            into xarray DataSets, such that 
            u,v,chs are the data arrays
            x,y are coordinates and 
            'x','y' are dimensions
            'dt' is a float that becomes an attribute
            'l_units' are units of length, string converted to an 
            attribute
            't_units' are time units (either 'dt' or 's'), 
            str->xr.attr
            'rows','cols' are additional attributes which are 
            just data.shape[0,1]
           
        """
        self._obj = xarray_obj
        self._average = None # not initialized

    @property
    def average(self):
        """ Return the mean flow field ."""
        self._average = self._obj.mean(dim='t')
        return self._average
    
    def crop(self,crop_vector = [None, None, None, None]):
        """ crop number of rows, cols from either side of the vector fields 
        Input: 
            self : xarray Dataset 
            crop_vector : [xmin,xmax,ymin,ymax] is a list of values crop the data, 
            defaults are None
        Return: 
            same object as the input    
        """
        xmin,xmax,ymin,ymax = crop_vector
        
        xmin = self._obj.x.min() if xmin is None else xmin
        xmax = self._obj.x.max() if xmax is None else xmax
        ymin = self._obj.y.min() if ymin is None else ymin
        ymax = self._obj.y.max() if ymax is None else ymax        
        
        self._obj = self._obj.sel(x=slice(xmin, xmax),y=slice(ymin,ymax))

        return self._obj

        

    def pan(self,dx=0.0,dy=0.0):
        """ moves the field by dx,dy in the same units as x,y """
        self._obj['x'] += dx
        self._obj['y'] += dy
        return self._obj

    def filterf(self):
        """Gaussian filtering of velocity """
        from scipy.ndimage.filters import gaussian_filter as gf
        self._obj['u'] = xr.DataArray(gf(self._obj['u'],1),dims=('x','y'))
        self._obj['v'] = xr.DataArray(gf(self._obj['v'],1),dims=('x','y'))
        return self._obj

    def __add__(self,other):
        """ add two datasets means that we sum up the velocities, assume
        that x,y,t,dt are all identical 
        """
        self._obj['u'] += other._obj['u']
        self._obj['v'] += other._obj['v']
        return self._obj

    def __sub__(self,other):
        """ add two datasets means that we sum up the velocities, assume
        that x,y,t,dt are all identical 
        """
        self._obj['u'] -= other._obj['u']
        self._obj['v'] -= other._obj['v']
        return self._obj
    
    
    def vorticity(self):
        """ calculates vorticity of the data array (at one time instance) 
        
        Input: 
            xarray with the variables u,v and dimensions x,y
        
        Output:
            xarray with the estimated vorticity as a scalar field with same dimensions
        
        """
        
        ux,_ = np.gradient(self._obj['u'],self._obj['x'],self._obj['y'],axis=(0,1))
        _,vy = np.gradient(self._obj['v'],self._obj['x'],self._obj['y'],axis=(0,1))
        # self._obj['w'] = xr.DataArray(vy - ux, dims=['x', 'y'])
        self._obj['w'] = xr.DataArray(vy - ux, dims=['x', 'y','t'])
        return self._obj
    
    def shear(self):
        """ calculates shear of the data array (single frame) 
        
        Input: 
            xarray with the variables u,v and dimensions x,y
        
        Output:
            xarray with the estimated shear as a scalar field data['w']
        
        """
        
        ux,_ = np.gradient(self._obj['u'],self._obj['x'],self._obj['y'],axis=(0,1))
        _,vy = np.gradient(self._obj['v'],self._obj['x'],self._obj['y'],axis=(0,1))
        # self._obj['w'] = xr.DataArray(vy - ux, dims=['x', 'y'])
        self._obj['w'] = xr.DataArray(vy + ux, dims=['x', 'y','t'])
        return self._obj

    def acceleration(self):
        """ calculates material derivative or acceleration of the data array (single frame) 
        
        Input: 
            xarray with the variables u,v and dimensions x,y
        
        Output:
            xarray with the estimated acceleration as a scalar field data['w']
        
        """
        ux,uy = np.gradient(self._obj['u'],self._obj['x'],self._obj['y'],axis=(0,1))
        vx,vy = np.gradient(self._obj['v'],self._obj['x'],self._obj['y'],axis=(0,1))
        
        ax = self._obj['u']*ux + self._obj['v']*uy
        ay = self._obj['u']*vx + self._obj['v']*vy

        self._obj['w'] = xr.DataArray(np.sqrt(ax**2+ay**2), dims=['x', 'y','t'])
        return self._obj
        
    def tke(self):
        """ estimates turbulent kinetic energy """
        self._obj['w'] = self._obj['u']**2 + self._obj['v']**2
        return self._obj
        
    def vec2scal(self, property='curl'):
        """ creates a dataset of scalar values on the same 
        dimensions and coordinates as the vector dataset
        Agruments:
            data : xarray.DataSet with u,v on t,x,y grid
        Returns:
            scalar_data : xarray.Dataset w on t,x,y grid
            'w' represents one of the following properties:
                - 'curl' or 'rot' - vorticity

        """
        # replace few common names
        property='vorticity' if property == 'curl' else property
        property = 'tke' if property == 'ken' else property
        
        method_name = str(property)
        method = getattr(self, method_name, lambda: "nothing")
        
        if len(self._obj.attrs['variables']) == 4: # only x,y,u,v
            self._obj.attrs['variables'].append(property)
        else:
            self._obj.attrs['variables'][-1] = property
            
        return method()

    def __mul__(self,scalar):
        '''
        multiplication of a velocity field by a scalar (simple scaling)
        '''
        self._obj['u'] *= scalar
        self._obj['v'] *= scalar

        return self._obj

    def __div__(self,scalar):
        '''
        multiplication of a velocity field by a scalar (simple scaling)
        '''
        self._obj['u'] /= scalar
        self._obj['v'] /= scalar

        return self._obj

    
    def set_dt(self,dt):
        self._obj.attrs['dt'] = dt
        
    def set_tUnits(self,tUnits):
        self._obj.attrs['tUnits'] = tUnits
        
    def rotate(self,theta=0.0):
        """ 
        use this method in order to rotate the data 
        by theta degrees in the clockwise direction
        in the present form of using xarray with coordinates
        it can only work for the cases with equal size along
        x and y 
        """
        
        theta = theta/360.0*2*np.pi
        
        xi = self._obj.x*np.cos(theta) + self._obj.y*np.sin(theta)
        eta = self._obj.y*np.cos(theta) - self._obj.x*np.sin(theta)
        Uxi = self._obj.u*np.cos(theta) + self._obj.v*np.sin(theta)
        Ueta = self._obj.v*np.cos(theta) - self._obj.u*np.sin(theta)
        
        self._obj['x'] = xi
        self._obj['y'] = eta
        self._obj['u'] = Uxi
        self._obj['v'] = Ueta
        
        
        if 'theta' in self._obj:
            self._obj['theta'] += theta
        else:
            self._obj['theta'] = theta
            
        return self._obj
        
    @property
    def get_dt(self):
        """ receives the dt from the set """
        return self._obj.attrs['dt']

    def spatial_filter(self, filter = 'median',size=(4,4),sigma=(1,1),mode='reflect'):
        """
        this method passes the velocity vectors U and V
        through a either a 4 x 4 median filter or a gaussian
        filter with sigma = 1
        Inputs: 
            filter : string, 'median' (default), 'gaussian'
            size  : (4,4) default, tuple of integers for 'median'
            sigma : float, for 'gaussian'
            mode : 'reflect' - see gaussian_filter for more info
        """    
        if filter is 'median':
            filt = lambda x: median_filter(x, size=size)
        elif filter is 'gaussian':
            filt = lambda x: gaussian_filter(x,sigma=sigma,mode=mode)
        else:
            raise ValueError('Wrong filter type: "median" or "gaussian"')
            
        for t in self._obj['t']:
            tmp = self._obj.sel(t=t)
            tmp['u'] = xr.DataArray(filt(tmp['u']),dims=['x','y'])
            tmp['v'] = xr.DataArray(filt(tmp['v']),dims=['x','y'])
            
        return self._obj
    
    def gaussian_smooth(self, scale=5, mask=False, mode='reflect'):
        """Apply gaussian kernel to convolution. Uses Scipy
           gaussian_filter method.
           Parameters:
           mode (str): {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
                       What to do at edges of matrix input. See Scipy docs
                       for details on what these do.
        """
        data = self._obj
        dims = _get_dims(data)

        sc_gaussian_nd = lambda data: gaussian_filter(data, scale, mode=mode)

        if mask:
            data_masked = data.where(data[mask_vars[dims]])
        else:
            data_masked = data.fillna(0.)

        return xr.apply_ufunc(sc_gaussian_nd, data_masked,
                              vectorize=True,
                              dask='parallelized',
                              input_core_dims = [dims],
                              output_core_dims = [dims],
                              output_dtypes=[data.dtype])

        

    # @property
    # def vel_units(self):
    #     " Return the geographic center point of this dataset."
    #     if self._vel_units is None:
    #         self._vel_units = self._obj.attrs.l_units + '/' + self._obj.attrs.t_units 
    #     return self._vel_units
        

class Vec:
    def __init__(self,x,y,u,v,chc,dt,lUnits='m',tUnits='s'):
        """ basic class that will hold the velocity field
            x,y are coordinates in lUnits
            u,v are velocity components in lUnits/tUnits
            CHC is some marker of bad vectors
            theta is not so clear but it's a global orientation angle, in degrees
            properties:
            lUnits, default is meter - 'm'
            tUnits, default is seconds - 's'
            velUnits, derived from  lUnits, tUnits
        """
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.chc = chc
        self.dt = dt
        self.lUnits = lUnits
        self.tUnits = tUnits
        self.velUnits = lUnits + '/' + tUnits
        self.theta = 0.0 # what is theta ? 
    
 
    
    def set_dt(self,dt):
        self.dt = dt
    
    def get_dt(self):
        return self.dt
    
    def set_Lunits(self,lUnits = 'm'):
        self.lUnits = lUnits # default is meter
    
    def set_tUnits(self,tUnits = 's'):
        self.tUnits = tUnits # default is sec
            

        
    def scale(self,resolution):
        """
        use this method to change the resolution of the 
        vector from [px/frame] to [m/sec] or any similar
        - resolution should be in [length/px]
        - time is generated from the original file and
          it is in microseconds
        """
        self.x = self.x*resolution
        self.y = self.y*resolution
        self.u = self.u*resolution/(self.dt*1e-6)
        self.v = self.v*resolution/(self.dt*1e-6)
        
    def move(self,dx,dy):
        """
        use this method to move the origin of the frame
        by dx and dy
        """
        self.x = self.x + dx
        self.y = self.y + dy

        
    def crop(self,xmin,xmax,ymin,ymax):
        """
        this method is used to crop a rectangular section 
        of the vector field defined as the region between 
        (xmin,ymin) and (xmax,ymax) 
        """
        temp = []
        indexes = []
        for i in range(len(self.x[:,0])):
            temp.append([])
            for j in range(len(self.x[0,:])):
                if self.x[i,j]<xmax and self.x[i,j]>xmin:
                    if self.y[i,j]<ymax and self.y[i,j]>ymin:
                        temp[-1].append((i,j))
        for i in temp:
            if len(i)>0:
                indexes.append(i)
        if len(indexes)==0:
            print('not valid crop values')
            return
        indexes = array(indexes)
        x, y = zeros(shape(indexes[:,:,0])), zeros(shape(indexes[:,:,0]))
        u, v = zeros(shape(indexes[:,:,0])), zeros(shape(indexes[:,:,0]))
        chc = zeros(shape(indexes[:,:,0]))
        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                x[i,j] = self.x[indexes[i,j][0],indexes[i,j][1]]
                y[i,j] = self.y[indexes[i,j][0],indexes[i,j][1]]
                u[i,j] = self.u[indexes[i,j][0],indexes[i,j][1]]
                v[i,j] = self.v[indexes[i,j][0],indexes[i,j][1]]
                chc[i,j] = self.chc[indexes[i,j][0],indexes[i,j][1]]
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.chc = chc
        
    def getVelStat(self):
        """
        this method calculates the data's ensamble mean and standard
        deviation values and assigns them to new atribtes
        of the instance vec.
        """
        u,v = self.u.flatten(), self.v.flatten()
        self.Umean, self.Ustd = norm.fit(u)
        self.Vmean, self.Vstd = norm.fit(v)
        
        
    def filterVelocity(self,filtr = 'med',size=(4,4)):
        """
        this method passes the velocity vectors U and V
        through a either a 4 x 4 median filter or a gaussian
        filter with sigma = 1
        Inputs: 
            filtr = 'med' (default), 'gauss', string
            size  = (4,4) default, tuple
        """
        if filtr == 'med':
            self.u = median_filter(self.u,size=size)
            self.v = median_filter(self.v,size=size)
        elif filtr == 'gauss':
            self.u = gaussian_filter(self.u,1)
            self.v = gaussian_filter(self.v,1)
        else:
            raise ValueError('Wrong filter. Choose Either "med" or "gauss"')
        
    
