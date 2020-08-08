import os
import glob
import numpy as np
from helpers import util
from data_preprocess.optical_flow import OpticalFlowExtractor
import multiprocessing

def main():
	
    data_path = '../data/lps_data/surveillance_camera'    
    data_selection_path = '../metadata/pain_no_pain_x2h_intervals_for_extraction.csv'
    out_dir_offsets = '../metadata/fixing_offsets_with_cam_on'
    out_file_final =  os.path.join(out_dir_offsets,'video_offsets_final.csv')

    fps = 10
    width = 672
    height = 380
    str_dir = '_'.join([str(val) for val in [width,height,fps]])
    out_dir_testing = '../data/pain_no_pain_x2h_intervals_for_extraction_'+str_dir+'fps'
    util.mkdir(out_dir_testing)


    ofe = OpticalFlowExtractor(output_dir = out_dir_testing, num_processes = multiprocessing.cpu_count(), output_flow = out_dir_testing)
    
    ofe.extract_frames(subjects_to_extract = ['aslan'])

if __name__=='__main__':
	main()