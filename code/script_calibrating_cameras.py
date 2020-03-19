import os
import glob
import numpy as np
from helpers import util, visualize
from multiview_frame_extractor import MultiViewFrameExtractor
import multiprocessing
import 

def extract_calibration_vid_frames():
	data_path = '../data/camera_calibration_videos'
	out_dir = '../data/camera_calibration_frames'
	data_selection_path = '../metadata/interval_calibration.csv'
	util.mkdir(out_dir)

	mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1/5., output_dir = out_dir,
                 views = [0,1,2,3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
	mve.extract_frames()

	# mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1/5., output_dir = out_dir,
 #                 views = [3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
	# mve.extract_frames(subjects_to_extract = ['cell1'])

def fix_filenames():
	pass
	# get video names

	# extract first frame

	# view first frames

	# what're the times i see

	# rename files



def main():
	
	# extract_calibration_vid_frames()

	out_dir_html = '../data/camera_calibration_frames'
	dirs_to_check = [os.path.join(out_dir_html, 'cell1','20200317130703_131800'), os.path.join(out_dir_html,'cell2','20200317130703_131800')]
	view_multiple_dirs(dirs_to_check, out_dir_html)

	# return
	# print ('hello')
	# data_path = '../data/lps_data/surveillance_camera'
	# out_dir = '../data/camera_calibration_frames'
	# data_selection_path = './interval_debug.csv'
	# util.mkdir(out_dir)

	# mve = MultiViewFrameExtractor(data_path = data_path, width= 396, height = 224, frame_rate = 1/5., output_dir = out_dir,
 #                 views = [0], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
	# mve.extract_frames(subjects_to_extract = ['sir_holger'])
	


if __name__=='__main__':
	main()
