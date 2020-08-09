import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import glob
import numpy as np
from helpers import util
from optical_flow import OpticalFlowExtractor
import multiprocessing

def main():
	
    mags = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps/aslan/20190103154700_165305/2_opt_mags.txt'
    mags = [float(val) for val in util.readLinesFromFile(mags)]
    mags = np.array(mags)
    print (np.min(mags), np.max(mags))
    for thresh in np.arange(0,1,0.1):
        less = np.sum(mags<thresh)/float(mags.size)*100
        more = 100-less
        print (thresh, less, more)

    # data_path = '../data/lps_data/surveillance_camera'    
    # data_selection_path = '../metadata/pain_no_pain_x2h_intervals_for_extraction.csv'
    # out_dir_offsets = '../metadata/fixing_offsets_with_cam_on'
    # out_file_final =  os.path.join(out_dir_offsets,'video_offsets_final.csv')

    # fps = 10
    # width = 672
    # height = 380
    # str_dir = '_'.join([str(val) for val in [width,height,fps]])
    # out_dir_testing = '../data/pain_no_pain_x2h_intervals_for_extraction_'+str_dir+'fps'
    # util.mkdir(out_dir_testing)


    # ofe = OpticalFlowExtractor(output_dir = out_dir_testing, num_processes = multiprocessing.cpu_count(), output_flow = out_dir_testing)
    
    # ofe.extract_frames(subjects_to_extract = ['aslan'])

if __name__=='__main__':
	main()