import os
import glob
import numpy as np
from helpers import util, visualize
from multiview_frame_extractor import MultiViewFrameExtractor
import multiprocessing
import glob
from scripts_for_visualization import view_multiple_dirs
import subprocess
import pandas as pd
import cv2

def extract_first_frames(vid_files, out_dir):
    # extract first frame
    for vid_file in vid_files:
        out_file = os.path.join(out_dir, os.path.split(vid_file)[1].replace('.mp4','.jpg'))
        command = ['ffmpeg', '-i', vid_file, '-y', '-vframes', '1', '-f', 'image2', out_file]
        subprocess.call(command)

    # view first frames
    visualize.writeHTMLForFolder(out_dir, height = 506, width = 896)
    
def main():
    print ('hello')
    data_path = '../data/lps_data/surveillance_camera'
    out_dir = '../scratch/check_first_frames'
    data_selection_path = '../metadata/intervals_for_extraction.csv'
    util.mkdir(out_dir)

    mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1., output_dir = out_dir,
                 views = [0,1,2,3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
    video_paths = mve.get_videos_containing_intervals()
    out_file = os.path.join('../metadata','intervals_for_extraction_video_file_list.txt')
    util.writeFile(out_file, video_paths)




if __name__=='__main__':
    main()