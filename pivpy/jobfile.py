# -*- coding: utf-8 -*-

import yaml
import run_jobfile

class jobfile:
    def __init__(self,images_path,pair_or_set,images_type,frames,window_size=[64,64],search_area_size=[128,128],overlap_per=50,dt=1,corr_type='fft',pre=None,post=None):
        self.images_path = images_path
        self.pair_or_set = pair_or_set
        self.images_type = images_type
        if self.pair_or_set == 'pair':
            if len(frames)==1:
                if frames.endswith(images_type):
                    self.frames = images_path+frames
                    self.frame_name = str(frames.split('.')[0])
                    self.image_files = 1
                else:
                    self.frames = images_path+frames+images_type
                    self.frame_name = str(frames)
                    self.image_files = 1
            else:
                if frames[0].endswith(images_type):    
                    self.frame_a = images_path+frames[0]
                    self.frame_name = str(frames[0].split('.')[0][:-1])
                else:
                    self.frame_a = images_path+frames[0]+images_type
                    self.frame_name = str(frames[0][:-1])
                if frames[1].endswith(images_type):
                    self.frame_b = images_path+frames[1]
                else:    
                    self.frame_b = images_path+frames[1]+images_type
                self.image_files = 2
                
        
        self.piv_paramters = {}
        self.piv_paramters['dt'] = dt
        self.piv_paramters['corr_type'] = corr_type
        self.piv_paramters['window_size'] = window_size
        self.piv_paramters['search_area_size']= search_area_size
        self.piv_paramters['overlap_per'] = overlap_per
        self.piv_paramters['preprocess'] = []
        self.piv_paramters['postprocess'] = []

        if type(pre) == str:
            self.piv_paramters['preprocess'].extend(pre.split(' '))
        elif type(pre) == list:
            self.piv_paramters['preprocess'].extend(pre)
        elif type(pre) == tuple:
            self.piv_paramters['preprocess'].extend(list(pre))
        
        if type(post) == str:
            self.piv_paramters['postprocess'].extend(post.split(' '))
        elif type(post) == list:
            self.piv_paramters['postprocess'].extend(post)
        elif type(post) == tuple:
            self.piv_paramters['postprocess'].extend(list(post))

        self.piv_paramters['preprocess_parameters']={}
        self.piv_paramters['postprocess_parameters']={}

        for preprop in self.piv_paramters['preprocess']:
            if str(preprop) == 'gausian_filter':
                print('cool')
            elif str(preprop) == 'wind_tunnel_mask2':
                self.piv_paramters['preprocess_parameters']['wind_tunnel_mask'] = {}
                same = input("Use same percentile for both images: ")
                if same == True or same == 'True' or str.lower(same)=='yes':
                    percentile_low = input("wind tunnel mask percentile low: ")
                    percentile_high = input("wind tunnel mask percentile high: ")
                    self.piv_paramters['preprocess_parameters']['wind_tunnel_mask']['percentile_a'] = (float(percentile_low),float(percentile_high))
                    self.piv_paramters['preprocess_parameters']['wind_tunnel_mask']['percentile_b'] = (float(percentile_low),float(percentile_high))
                else:
                    percentile_low_a = input("wind tunnel mask percentile low for first image: ")
                    percentile_high_a = input("wind tunnel mask percentile high for first image: ")
                    percentile_low_b = input("wind tunnel mask percentile low for second image: ")
                    percentile_high_b = input("wind tunnel mask percentile high for second image: ")
                    self.piv_paramters['preprocess_parameters']['wind_tunnel_mask']['percentile_a'] = (float(percentile_low_a),float(percentile_high_a))
                    self.piv_paramters['preprocess_parameters']['wind_tunnel_mask']['percentile_b'] = (float(percentile_low_b),float(percentile_high_b))

        for postprop in self.piv_paramters['postprocess']:
            if str(postprop) == 'max_mag_outlier':
                max_U = float(input("Max magnitude for U: "))
                max_V = float(input("Max magnitude for V: "))
                self.piv_paramters['postprocess_parameters']['max_mag_outlier'] = (max_U,max_V)
                
            elif str(postprop) == 'mark_outlier':
                hist_bins = int(input("number of histograms bins: "))
                Z_thresh = float(input("Z test threshold: "))
                q_thresh = float(input("q test threshold: "))
                self.piv_paramters['postprocess_parameters']['mark_outlier']=(hist_bins,Z_thresh,q_thresh)
            


            
    
'''
images_path = '/Users/User/Documents/University/Masters/Turb_Lab/PIV/PIV_challenge/2005/C/'
output_path = '/Users/User/Documents/University/Masters/Turb_Lab/PIV/PIV_challenge/2005/C/output1/'
pair_or_set = 'set'
images_type = 'bmp'
frames = ''
corr_type='fft'
window_size = [32,32]
search_area_size = [64,64]
overlap_per = 50
dt = 1
pixel_to_meter = 1
job1 = jobfile(images_path,pair_or_set,images_type,frames,window_size=window_size,search_area_size=search_area_size)
run_jobfile.run_jobfile(job1,output_path)

     
images_path = '/Users/User/Documents/University/Masters/Turb_Lab/wind_tunnel_data/Run3/'
output_path = '/Users/User/Documents/University/Masters/Turb_Lab/wind_tunnel_data/Run3/output_28_06_18/'
pair_or_set = 'set'
images_type = 'im7'
frames = ''
corr_type='fft'
window_size = [64,64]
search_area_size = [128,128]
overlap_per = 50
dt = 0.0005
pixel_to_meter = 0.000118689
job1 = jobfile(images_path,pair_or_set,images_type,frames,pre=['wind_tunnel_mask'],post=['max_mag_outlier'])
run_jobfile.run_jobfile(job1,output_path)
'''