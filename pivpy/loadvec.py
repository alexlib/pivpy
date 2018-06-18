#!/usr/bin/pythonw
""" LOADVEC - load a PIV .vec file onto a python oriented envierment 

addapted from Alex Liberzon's code.
extended by Ron Shnapp 24.5.15


"""
import os
#import numpy as np
#from numpy import *
from numpy import genfromtxt, meshgrid, where, zeros
#import matplotlib.pylab as mpl
#import matplotlib.pyplot as plt
from vecPy import Vec
from string import upper, lower


def get_dt(fname,path):
    """given a .vec file this will return the delta t 
    from the file in micro seconds"""
    # os.chdir(path) BUG
    fname = os.path.join(os.path.abspath(path),fname) # just make a full path name 
    # new way of opening and closing the file
    with open(fname) as f:
        header = f.readline()
        
    ind1 = header.find('MicrosecondsPerDeltaT')
    dt = float(header[ind1:].split('"')[1])
    return dt

def get_units(fname, path):
    """ given a .vec file this will return the names of length and velocity units """
    # os.chdir(path) BUG
    fname = os.path.join(os.path.abspath(path),fname) # just make a full path name 
    # new way of opening and closing the file
    with open(fname) as f:
        header = f.readline()
    
    ind2= header.find('VARIABLES=')
    ind3 = header.find('"X',ind2)
    ind4 = header.find('"',ind3+1)
    header[ind3:ind4+1]
    lUnits = header[ind3+3:ind4]
    # print lUnits

#     ind3 = header.find('"Y',ind2)
#     ind4 = header.find('"',ind3+1)
#     header[ind3:ind4+1]
#     lUnits = header[ind3+3:ind4]
#     print lUnits

    ind3 = header.find('"U',ind2)
    ind4 = header.find('"',ind3+1)
    header[ind3:ind4+1]
    velUnits = header[ind3+3:ind4]
    # print velUnits
    
#    tUnits = velUnits.split('/')[1]
    if velUnits == 'pixel':
        tUnits = 'dt'
    else:
        tUnits = velUnits.split('/')[1]
# 
#     ind3 = header.find('"V',ind2)
#     ind4 = header.find('"',ind3+1)
#     header[ind3:ind4+1]
#     velUnits = header[ind3+3:ind4]
#     print velUnits

    # fallback if nothing is read properly
    if lUnits is None:
        lUnits = 'mm'
    if velUnits is None:
        velUnits = 'm/s'
    if tUnits is None:
        tUnits = 's'
    
    return lUnits, velUnits, tUnits


def get_data(fname,path):
    """this function gathers and retuens the data found in
    a single .vec file"""
    fname = os.path.join(os.path.abspath(path),fname) # just make a full path name 
    if fname.lower().endswith('.vec'):
        data = genfromtxt(fname,skip_header=1,delimiter=',',usecols=(0,1,2,3,4))
    else:
        raise 'Wrong file extension'
        
    return data
    
def get_data_openpiv_txt(fname,path):
    """this function gathers and retuens the data found in
    a single .txt file created by OpenPIV"""
    fname = os.path.join(os.path.abspath(path),fname)
    if os.path.isfile(fname) and fname.endswith('.txt'): 
        data = genfromtxt(fname,usecols=(0,1,2,3,4))
    else:
        raise ValueError('Wrong file or file extension')
    
    return data
    
def get_data_openpiv_vec(fname,path):
    """this function gathers and retuens the data found in
    a single .vec file created by OpenPIV"""
    fname = os.path.join(os.path.abspath(path),fname)
    if os.path.isfile(fname) and fname.endswith('.vec'): 
        data = genfromtxt(fname,usecols=(0,1,2,3,4))
    else:
        raise ValueError('Wrong file or file extension')
    
    return data
	
def read_directory(dirname, ext='vec'):
    # list_files = os.listdir(dirname)
    list_files = [s for s in os.listdir(dirname) if s.rpartition('.')[2] in (lower(ext),upper(ext))]
    return list_files
	
def patternize(lst):
    """helper function for vecToMatrix"""
    lst = sorted(lst)
    n = [lst[0]]
    for i in range(len(lst)-1):
        if lst[i] != lst[i+1]:
            n.append(lst[i+1])    
    return n
    
def vecToMatrix(data):
    """ helper finction for vecToVec.
    this function takes vector form data and shifts it in
    to a matrix form. return is 4 matrices:
    X,Y - x and y axis position matrices in meshgrid form
    U,V - velocity of the flow"""
    x = patternize(data[:,0])
    y = patternize(data[:,1])
    X,Y = meshgrid(x,y)
    U,V,CHC = zeros(X.shape), zeros(X.shape), zeros(X.shape)
    for row in range(len(data[:,0])):
        x,y,u,v,chc = data[row,:]
        j,i = where(X==x)[1][0], where(Y==y)[0][0]
        U[i,j],V[i,j],CHC[i,j] = u,v,chc
    return (X,Y,U,V,CHC)

def vecToVec(fname,path):
    """ 
    generate directly the vec class object from a vec file
    """
    X,Y,U,V,CHC = vecToMatrix(get_data(fname,path))
    dt = get_dt(fname,path)
    lUnits, velUnits, tUnits = get_units(fname,path)
    vector = Vec(X,Y,U,V,CHC,dt,lUnits=lUnits,tUnits=tUnits)
    return vector
    
        
def getVecList(path, fnames = None, resolution=1, 
               crop=False, rotate=False,Filter=True):
    """
    this function returns a list of vec instances from .vec files. Use this 
    if you want to analize a bunch of vec files together
    
    inputs - 
    
    path (string) - path to the directory that holds the data
    fnames (None or list of strings) - if None, returns all the .vec files in
           path. If list will return vec instances for the data in each file
           name
    """
    if fnames == None:
        fnames = os.listdir(path)  
    else:
        if type(fnames) != list:
            raise TypeError('fnames should be a list of strings, of None')

    lst = []
    for n in fnames:
        if '.vec' in n:
            X,Y,U,V,CHC = vecToMatrix(get_data(n,path))
            dt = get_dt(n,path)
            vector = Vec(X,Y,U,V,CHC,dt)
            vector.scale(resolution)
            if Filter: vector.filterVelocity('med',5)
            if rotate: vector.rotate(rotate)
            if crop: vector.crop(crop[0],crop[1],crop[2],crop[3])
            lst.append(vector)
    return lst
     
def readTimeStamp(fname,path):
    """reads an insight tstmp file and returns
    an array of the times at which photos were
    taken at relative to the begining of
    aquasition"""
    fname = os.path.join(os.path.abspath(path),fname)
    num_lines = sum(1 for line in open(fname))
    f = open(fname)
    for i in range(3):
        f.readline()
    strt = [f.readline().split()[1] for i in range(num_lines-4)]
    t = [float(i)/1000000 for i in strt]
    return t