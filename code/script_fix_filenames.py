import os
import glob
import numpy as np
from helpers import util, visualize
from multiview_frame_extractor import MultiViewFrameExtractor
import multiprocessing
import glob
import subprocess
import pytesseract as ocr
import cv2
import re
import pandas as pd

def extract_first_frames(vid_files, out_dir):
    # extract first frame
    for vid_file in vid_files:
        out_file = os.path.join(out_dir, os.path.split(vid_file)[1].replace('.mp4','.jpg'))
        # if not os.path.exists(out_file):
        command = ['ffmpeg', '-i', vid_file, '-y', '-vframes', '1', '-f', 'image2', out_file]
        subprocess.call(command)
        
    # view first frames
    visualize.writeHTMLForFolder(out_dir, height = 506, width = 896)
    
def get_vid_time_from_frame(im_file):
    
    im = cv2.imread(im_file)
    # converting to rgb from bgr
    im = im[:,:,::-1]
    # clipping the image for speed
    im = im[:200,:1000]
    # look for text
    str_found = ocr.image_to_string(im)

    # match the time string
    str_found = str_found.strip()
    pattern = re.compile('[0-9][0-9]:[0-9][0-9]:[0-9][0-9]')
    match = re.search('[0-9][0-9]:[0-9][0-9]:[0-9][0-9]',str_found)

    # if found, return video name time and frame text time as ints
    if match:
        str_found = match.group(0)
        vid_name_st = int(im_file[-10:-4])
        vid_frame_st = int(str_found[-8:].replace(':',''))
        print ([vid_name_st,vid_frame_st])
        return [vid_name_st,vid_frame_st]

    # else return none
    return None    

def check_diff_reasonable(row,thresh):
    keep = None
    try:
        [vid_time,frame_time] = [pd.to_datetime('%06d'%val, format='%H%M%S') for val in row]
        if vid_time>frame_time:
            diff = vid_time - frame_time
            diff = int(diff.total_seconds())
        else:
            diff = frame_time - vid_time
            diff = -1*int(diff.total_seconds())
        if np.abs(diff)<=thresh:
            keep = diff
    except:
        pass
    
    return keep

def sort_auto_and_manual_checks(im_files,times_arr,out_file_auto, out_file_manual_check ):
    # times_arr = np.load(times_file)
    times_arr_idx = list(times_arr[:,0])

    im_check =[]
    im_auto = ['im_file,offset']

    # convert to time stamps
    for idx_im_file, im_file in enumerate(im_files):
        if idx_im_file in times_arr_idx:
            
            idx = times_arr_idx.index(idx_im_file)
            row = times_arr[idx]
            assert row[0]==idx_im_file

            diff_val = check_diff_reasonable(row[1:],3)
            if diff_val is not None:
                im_auto.append(im_file+','+str(diff_val))
            else:
                im_check.append(im_file)
        else:
            im_check.append(im_file)

    print (im_check)
    print (im_auto)

    print (len(im_check))
    print (len(im_auto))

    assert (len(im_check)+len(im_auto)-1==len(im_files))
    
    util.writeFile(out_file_auto,im_auto)
    util.writeFile(out_file_manual_check, im_check)

def main():
    data_path = '../data/lps_data/surveillance_camera'
    
    data_selection_path = '../metadata/intervals_for_extraction.csv'
    out_file = os.path.join('../metadata','intervals_for_extraction_video_file_list.txt')
    out_file_offsets_auto = '../metadata/video_offsets_auto.csv'
    out_file_offsets_manual = '../metadata/video_offsets_manual.csv'
    out_file_offsets_all = '../metadata/video_offsets_all.csv'

    out_dir = '../scratch/check_first_frames'
    times_file = os.path.join(out_dir,'times.npy')
    out_file_manual_check = os.path.join(out_dir,'manual_check.txt')
    im_list_file = os.path.join(out_dir,'im_list.txt')
    util.mkdir(out_dir)

    ## step 1 - get videos needed for our intervals

    # mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1., output_dir = out_dir,
    #              views = [0,1,2,3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
    # video_paths = mve.get_videos_containing_intervals()
    # util.writeFile(out_file, video_paths)

    ## step 2 - extract their first frames. make an im list file

    # vid_files = util.readLinesFromFile(out_file)
    # vid_files = list(set(vid_files))
    # print (len(vid_files))
    # extract_first_frames(vid_files, out_dir)
    # im_files = glob.glob(os.path.join(out_dir,'*.jpg'))
    # im_files.sort()
    # print (len(im_files))
    # util.writeFile(im_list_file, im_files)

    ## step 3 - do ocr to automatically get first frame times. save time array
    ## time np array is nx3. each row is [file idx, video time (int), frame time (int)]

    # im_files = util.readLinesFromFile(im_list_file)
    # arr = []    
    # for idx_im_file, im_file in enumerate(im_files):
    #     times = get_vid_time_from_frame(im_file)
    #     if times is not None:
    #         arr.append([idx_im_file]+times)
    # arr = np.array(arr)
    # print (arr.shape)
    # np.save(times_file,arr)


    ## step 4 - make list of files for manual checking
    ## save auto offsets as csv

    # im_files = util.readLinesFromFile(im_list_file)
    # times_arr = np.load(times_file)
    # sort_auto_and_manual_checks(im_files, times_arr, out_file_offsets_auto, out_file_manual_check)

    ## step 5 - do the manual checking . code in script_fix_filenames_local
    ## results of manual offsets are also in a csv now

    ## step 6 - merge offsets. 
    # auto_offsets = pd.read_csv(out_file_offsets_auto)
    # manual_offsets =  pd.read_csv(out_file_offsets_manual)
    # offsets = pd.concat([auto_offsets, manual_offsets], axis =0).reset_index()
    # offsets.to_csv(out_file_offsets_all,columns = ['im_file','offset'], index = False)


    ## step 7 - double check the offsets
    ## makes an md with vid time, frame time image, and offset on each row
    offsets = pd.read_csv(out_file_offsets_all)
    out_dir_check = '../scratch/check_times'
    md_file = os.path.join(out_dir_check,'double_check.md')
    util.mkdir(out_dir_check)
    md_rows =[]
    md_rows.append('**Just Checking**')
    md_rows.append(' ')
    md_rows.append('row idx|vid_name|vid time|frame time|offset')
    md_rows.append(':---:|:---:|:---:|:---:|:---:')
    # md_rows.append(' ')
    for idx_row, row in offsets.iterrows():
        im_path = row['im_file']
        offset = row['offset']
        vid_time = os.path.split(im_path)[1][-10:-4]
        out_file = os.path.join(out_dir_check, os.path.split(im_path)[1])
        
        if not os.path.exists(out_file):
            im = cv2.imread(im_path)[:200,:1000]
            cv2.imwrite(out_file, im)
        str_arr = [str(idx_row),os.path.split(im_path)[1][:-4],vid_time,'![]('+os.path.split(out_file)[1]+')',str(offset)]
        md_rows.append('|'.join(str_arr))
    util.writeFile(md_file, md_rows)







if __name__=='__main__':
    main()