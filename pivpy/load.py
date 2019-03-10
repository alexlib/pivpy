import os
import numpy as np
import scipy
from data_structures import  *
if 'jobfile.py' in os.listdir():
    from jobfile import *
from read_vec_file import *
import yaml

def check_str_bool(s):
    return s in ['True' ,'true', '1', 't', 'y','YES' ,'Yes','yes', 'yeah','Yeah', 'yup', 'certainly', 'uh-huh']

def field_data_properties(source,ws,frame,time,images_path,time_unit,time_scale_to_seconds,length_unit,length_scale_to_meter):
    vec_prop = vec_properties(str(source),str(ws),str(time_unit),str(time_scale_to_seconds),str(length_unit),str(length_scale_to_meter))
    field_prop = field_properties(str(frame),str(time),str(images_path),str(source),str(time_unit),str(time_scale_to_seconds),str(length_unit),str(length_scale_to_meter))

    return vec_prop,field_prop

def load_vec1(x,y,u,v,s2n,vec_prop):
    vec = vector(x,y,u,v,s2n,vec_prop)
    return vec

def data_processing(X,Y,U,V,S2N):
    if type(X) == np.ndarray and type(Y) == np.ndarray and type(U) == np.ndarray and type(V) == np.ndarray:
        if len(X.shape)==1 and len(Y.shape)==1 and len(U.shape)==1 and len(V.shape)==1:
            pass
        elif len(X.shape)==1 and len(Y.shape)==1 and len(U.shape)==2 and len(V.shape)==2:
            U = U.flatten()
            V = V.flatten()
            if S2N is not None:
                S2N = S2N.flatten()
        else:
            X = X.flatten()
            Y = Y.flatten()
            U = U.flatten()
            V = V.flatten()
            if S2N is not None:
                S2N = S2N.flatten()

    elif type(X) == list and type(Y) == list and type(U) == np.ndarray and type(V) == np.ndarray:
        X = np.array(X)
        Y = np.array(Y)
    
    return X,Y,U,V,S2N

def load_field1(X,Y,U,V,S2N,field_prop,vec_prop,source=None,ws=None,frame=None,time=None,images_path=None,time_unit=None,time_scale_to_seconds=None,length_unit=None,length_scale_to_meter=None):
    #data proccessing
    X,Y,U,V,S2N = data_processing(X,Y,U,V,S2N)
    
    if field_prop is None or vec_prop is None:
    
        if source==None or source=='':
            source = input("Algorithm source: ")
        
        if ws==None or ws == '' or ws == []:
            ws_x = input("window size in the X direction: ")
            ws_y = input("window size in the Y direction: ")
            ws = [ws_x,ws_y]
        
        if frame==None or frame == '' or frame == []:
            frame = input("Frame name: ")
            
        if images_path==None or images_path == '':
            images_path = input("images directory: ")
            
        if time_scale_to_seconds==None or time_scale_to_seconds == '':
            time_scale_to_seconds = float(input("dt in seconds: "))
        
        if length_scale_to_meter==None or length_scale_to_meter == '':
            length_scale_to_meter = float(input("length scale to meter: "))
        
        vec_prop,field_prop = field_data_properties(source,ws,frame,time,images_path,time_unit,time_scale_to_seconds,length_unit,length_scale_to_meter)


    field1 = field(field_prop)
    for ind in range(len(X)):
        vec = load_vec1(X[ind],Y[ind],U[ind],V[ind],S2N[ind],vec_prop)
        field1.add_vec(vec)
    return field1

def load_field2(X,Y,U,V,S2N,frame_name):
    X,Y,U,V,S2N = data_processing(X,Y,U,V,S2N)
    vec_prop,field_prop = field_data_properties('','',str(frame_name),'','','','','','')
    field1 = field(field_prop)
    for ind in range(len(X)):
        if S2N is not None:
            vec = load_vec1(X[ind],Y[ind],U[ind],V[ind],S2N[ind],vec_prop)
        else:
            vec = load_vec1(X[ind],Y[ind],U[ind],V[ind],0,vec_prop)
        field1.add_vec(vec)
    return field1

def parse_jobfile(jobfile):
    source = 'Tomers PIV algorithm'
    ws = jobfile.piv_paramters['window_size']
    if jobfile.pair_or_set.lower() == 'pair':
        if jobfile.image_files == 1:
            frame = jobfile.frame_name
        else:
            frame = [jobfile.frame_a,jobfile.frame_b]
    elif jobfile.pair_or_set.lower() == 'set':
        frame = jobfile.frame_name
    
    images_path = jobfile.images_path
    time_unit = 'dt'
    time_scale_to_seconds = jobfile.piv_paramters['dt']
    length_unit = 'pixel'
    time = ''
    length_scale_to_meter=''

    return field_data_properties(source,ws,frame,time,images_path,time_unit,time_scale_to_seconds,length_unit,length_scale_to_meter)
    

def check_for_yaml(file_name,path):
    if os.path.isfile(path+str(file_name).split('.')[0]+'.yaml'):
        with open(path+str(file_name).split('.')[0]+'.yaml', 'r') as stream:
            try:
                job1 = yaml.load(stream)
                return parse_jobfile(job1)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        return None
  
def load_field_from_vec(file_name,path,field_prop=None,vec_prop=None):
    file_name = str(file_name)
    if file_name.endswith('.vec'):
        X,Y,U,V  = read_vec_file(path+file_name)
        frame_name = file_name.split('.')[0]
    else:
        X,Y,U,V  = read_vec_file(path+file_name+'.vec')
        frame_name = file_name
    yaml_data = check_for_yaml(file_name,path)
    
    if yaml_data is not None:
        field1 = load_field2(X,Y,U,V,None,str(frame_name))
    elif field_prop==None and vec_prop==None:
        field1 = load_field2(X,Y,U,V,None,str(frame_name))
    elif field_prop==None and vec_prop is not None:
        field1 = load_field1(X,Y,U,V,None,field_prop,vec_prop)
    else:
        field1 = load_field1(X,Y,U,V,None,field_prop,vec_prop)
    return field1

def load_field_from_npz(file_name,path,field_prop=None,vec_prop=None):
    file_name = str(file_name)
    if file_name.endswith('.npz'):
        data  = np.load(path+file_name)
    else:
        data  = np.load(path+file_name+'.npz')
    yaml_data = check_for_yaml(file_name,path)
    
    X = data['X']
    Y = data['Y']
    U = data['U']
    V = data['V']
    S2N = data['S2N']

    if yaml_data is not None:
        field1 = load_field1(X,Y,U,V,S2N,yaml_data[1],yaml_data[0])
    else:
        field1 = load_field1(X,Y,U,V,S2N,field_prop,vec_prop)

    return field1

def load_npz_run_from_directory(directory_path,type_ending):
    run1 = run()
    files = [ fname for fname in os.listdir(directory_path) if fname.endswith(type_ending+'.npz')]
    for file in files:
        field1 = load_field_from_npz(file,directory_path,None,None)
        run1.add_field(field1)
    return run1

def load_vec_run_from_directory(directory_path,type_ending):
    run1 = run()
    files = [ fname for fname in os.listdir(directory_path) if fname.endswith(type_ending+'.vec')]
    run_input = check_str_bool(input('Do you want to input parameters of run?'))
    con_run = check_str_bool(input('Are the parameters of the run constant?'))
    if run_input and con_run:
        source = str(input('Source of data:'))
        ws = str(input('Window size of algorithm:'))
        enum_frame = check_str_bool(input('Use file names as frame names?'))
        time_unit= str(input('Time units:'))
        time_scale_to_seconds= str(input('Time units to seconds:'))
        length_unit= str(input('Length units:'))
        length_scale_to_meter= str(input('Length units to meters:'))
        if enum_frame:
            for file in files:
                vec_prop,field_prop = field_data_properties(source,ws,file,'','',time_unit,time_scale_to_seconds,length_unit,length_scale_to_meter)
                field1 = load_field_from_vec(file,directory_path,field_prop,vec_prop)
                run1.add_field(field1)
            return run1
        else:
            for i in range(len(files)):
                vec_prop,field_prop = field_data_properties(source,ws,i,'','',time_unit,time_scale_to_seconds,length_unit,length_scale_to_meter)
                field1 = load_field_from_vec(files[i],directory_path,field_prop,vec_prop)
                run1.add_field(field1)
    if run_input and not con_run:
        for file in files:
            vec_prop,field_prop = field_data_properties('','','','','','','','','')
            field1 = load_field_from_vec(file,directory_path,None,vec_prop)
            run1.add_field(field1)
        return run1
    else:
        for file in files:
            field1 = load_field_from_vec(file,directory_path)
            run1.add_field(field1)
        return run1
        
