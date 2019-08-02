import os
from helpers import util, visualize
from helpers import parse_motion_file as pms
import cv2
import numpy as np
import glob
from datetime import datetime
import pandas as pd
import subprocess

def extract_frame_at_time(video_file, out_dir, time_curr):
	
	# for time_curr in vid_times:
	timestampStr = pd.to_datetime(time_curr).strftime('%H:%M:%S')
	timestampStr_ = pd.to_datetime(time_curr).strftime('%H_%M_%S')
	command = ['ffmpeg', '-ss']
	command.append(timestampStr)
	command.extend(['-i',video_file])
	command.extend(['-vframes 1'])
	command.append('-vf scale=448:256')
	command.append('-y')
	command.append(os.path.join(out_dir,'frame_'+timestampStr_+'.jpg'))
	command.append('-hide_banner')

	command = ' '.join(command)
	return command

	# ffmpeg -ss 01:23:45 -i input -vf scale=448:256 ch06_20181209093714/frame_%09d.jpg -hide_banner


def main():
	in_dir = '../data/tester_kit/naughty_but_nice'
	motion_files = glob.glob(os.path.join(in_dir,'*.txt'))
	motion_files.sort()

	print (motion_files)
	# input()

	out_dir_meta = os.path.join('../scratch','frames_at_motion')
	util.mkdir(out_dir_meta)

	for motion_file in motion_files:
		print (motion_file)
		video_file = motion_file.replace('.txt','.mp4')
		vid_name = os.path.split(motion_file)[1]
		vid_name = vid_name[:vid_name.rindex('.')]
		
		out_dir = os.path.join(out_dir_meta, vid_name)
		util.mkdir(out_dir)

		cam_time = os.path.split(motion_file)[1]
		cam_time = cam_time[:cam_time.rindex('.')]
		cam_time = cam_time.split('_')
		cam = int(cam_time[0][2:])
		vid_start_time = datetime.strptime(cam_time[1],'%Y%m%d%H%M%S')
		vid_start_time = np.datetime64(vid_start_time)
		print (type(vid_start_time))

		df = pms.read_motion_file(motion_file)
		



		# for cam_num in [3,8]:
			# camera_str = 'D'+str(cam_num)
		motion_str = 'Motion Detection Started'
		rows = df.loc[(df['Minor Type']==motion_str),['Date Time']]
		
		motion_times =rows.iloc[:,0].values
		vid_times = motion_times - vid_start_time
		vid_times = vid_times.astype(np.datetime64)
		print (vid_start_time)
		print (motion_times[:5])
		print (vid_times[:5])



		# for time_curr in vid_times:
		commands= [extract_frame_at_time(video_file, out_dir, time_curr) for time_curr in vid_times]
		print (len(commands))
		commands = set(commands)
		print (len(commands))
		commands_file =os.path.join(out_dir,'commands.txt') 
		util.writeFile(commands_file, commands)
		print (commands_file)

		for command in commands:
			subprocess.call(command, shell=True)

		visualize.writeHTMLForFolder(out_dir, height=256, width = 448)
		# break

		# input()
		# for r in rows:
		# 	print (r)
			# diff = r['Date Time'] - vid_start_time
			# print (vid_start_time, r['Date Time'], diff)

		# break

		

if __name__=='__main__':
	main()
