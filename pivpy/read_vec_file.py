# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def read_vec_file(file):
    vecfile = open(file,'rb')
    a = vecfile.readlines()
    title = a[0].decode("utf-8")
    num_of_rows = len(a)-1
    X_pixel = np.zeros((num_of_rows,1))
    Y_pixel = np.zeros((num_of_rows,1))
    U_pixel = np.zeros((num_of_rows,1))
    V_pixel = np.zeros((num_of_rows,1))
    for i in range(num_of_rows):
        row = a[i+1].decode("utf-8").split(',')
        X_pixel[i] = float(row[0])
        Y_pixel[i] = float(row[1])
        U_pixel[i] = float(row[2])
        V_pixel[i] = float(row[3])
        
    return X_pixel,Y_pixel,U_pixel,V_pixel        
            

'''
f = '/Users/User/Documents/University/Masters/Turb_Lab/PIV/piv_algorithm/Compare/Case_A/insight/Case_a_insight.vec'        
X_pixel,Y_pixel,U_pixel,V_pixel = read_vec_file(f)
'''