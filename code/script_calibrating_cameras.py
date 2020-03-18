import os
import glob
import numpy as np
from helpers import util, visualize
from multiview_frame_extractor import MultiViewFrameExtractor
import multiprocessing

def extract_calibration_vid_frames():
	pass

def main():
	print ('hello')
	data_path = '../data/camera_calibration_videos'
	out_dir = '../data/camera_calibration_frames'
	data_selection_path = '../metadata/interval_calibration.csv'
	util.mkdir(out_dir)

	mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1/5., output_dir = out_dir,
                 views = [0,2,3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
	mve.extract_frames()
	


if __name__=='__main__':
	main()
