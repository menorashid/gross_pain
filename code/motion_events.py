import os
from helpers import util, visualize
from data_frames import Motion_File_Data
import numpy as np
import glob


def main():
	dir_meta = '../data/surveillance_camera/Clear_Pain_No_Pain'
	horse_dirs = [dir_curr for dir_curr in glob.glob(os.path.join(dir_meta,'*')) if os.path.isdir(dir_curr)]
	
	motion_counts = [[],[]]
	sub_dirs = ['Pain','No_Pain']
	motion_str = 'Motion Detection Started'
	for horse_dir in horse_dirs:
		for idx_sub_dir,sub_dir in enumerate(sub_dirs):
			dir_curr = os.path.join(horse_dir, sub_dir)
			motion_files = glob.glob(os.path.join(dir_curr,'*.txt'))
			for motion_file in motion_files:
				print (motion_file)
				motion_data = Motion_File_Data(motion_file)
				motion_times = motion_data.get_motion_times(motion_str)
				motion_counts[idx_sub_dir].append(motion_times.size)
				
	print ('Type','Count','Min','Max','Mean','Std')
	for idx, motion_count_curr in enumerate(motion_counts):
		motion_count_curr = np.array(motion_count_curr)
		print (sub_dirs[idx],motion_count_curr.size, np.min(motion_count_curr), np.max(motion_count_curr), np.mean(motion_count_curr), np.std(motion_count_curr))



if __name__=='__main__':
	main()
