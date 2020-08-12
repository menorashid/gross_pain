import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread, imsave
from helpers import util, visualize
import glob
import multiprocessing
NUM_CAMERAS = 8

def get_all_jpg_paths(frame_folder):
    paths = []
    for root, dirs, files in os.walk(frame_folder):
        for name in files:
            if name.endswith((".jpg")):
                path = os.path.join(root, name)
                paths.append(path)
    return paths


def get_camera_from_path(path):
    split = re.split('/', path)
    subject = split[3]
    view = split[5]
    lookup_viewpoint = pd.read_csv('../metadata/viewpoints.csv', index_col='subject')
    camera = lookup_viewpoint.at[subject, view]
    return camera
    
def script_median_of_all():

    
    # frame_folder = '../data/intervals_for_extraction_128_128_0.1fps/'
    # out_folder = '../data/median_bg'
    # skip_num = 1

    frame_folder = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps/'
    out_folder = '../data/testing_median'
    skip_num = 100
    util.mkdir(out_folder)

    path_list = get_all_jpg_paths(frame_folder)
    path_list = path_list[::skip_num]
    per_cam_lists = []

    # Make list of empty lists, one per camera
    for i in range(NUM_CAMERAS):
        per_cam_list = []
        per_cam_lists.append(per_cam_list)
        
    # Read and sort images into the right camera list
    with tqdm(total=len(path_list)) as pbar:
        for idx, path in enumerate(path_list):
            pbar.update(1)
            camera = int(get_camera_from_path(path))
            cam_idx = camera-1  # Make zero-indexed
            img = imread(path)
            per_cam_lists[cam_idx].append(img)

    # Per camera: convert to array, compute median and save
    with tqdm(total=NUM_CAMERAS) as pbar:
        for i in range(NUM_CAMERAS):
            pbar.update(1)
            ar = np.asarray(per_cam_lists[i])
            med = np.median(ar, axis=0)
            imsave(os.path.join(out_folder,'median_0.1fps_camera_{}.jpg'.format(i)), med.astype('uint8'))


def get_median_from_dir(arg):
    (dir_curr, step, min_val, out_file) = arg
    img_files = glob.glob(os.path.join(dir_curr,'*.jpg'))
    img_files.sort()
    num_files= len(img_files)
    step = max(1,num_files//min_val)
    to_keep = img_files[::step]

    print (len(to_keep))
    im_list =[]
    for path in to_keep:
        im_list.append(imread(path))
        
    ar = np.asarray(im_list)
    med = np.median(ar, axis=0)
    imsave(out_file, med.astype('uint8'))
    
    # print (len(img_files))
    # print (len(view_paths))
    # print (view_paths[:10])

def rejected_idea():
    data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps'
    interval_paths = glob.glob(os.path.join(data_path,'*','*'))
    
    print (len(interval_paths), interval_paths[:3])
    views = range(4)
    step = 10
    min_val = 1000

    # lookup_viewpoint = pd.read_csv('../metadata/viewpoints.csv', index_col='subject')
    # camera = lookup_viewpoint.at[subject, view]

    args = []
    for interval_path in interval_paths:
        for view in views:
            dir_curr = os.path.join(interval_path, str(view))
            assert os.path.exists(dir_curr)
            out_file = dir_curr+'_bg.jpg'
            # print (out_file)
            args.append((dir_curr, step, min_val, out_file))
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(get_median_from_dir, args)
    pool.close()
    pool.join()

def main():
    data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps'
    out_folder = '../data/bg_per_month_672_380_0.2fps'
    util.mkdir(out_folder)
    interval_paths = glob.glob(os.path.join(data_path,'*','*'))
    interval_paths = [dir_curr for dir_curr in interval_paths if os.path.isdir(dir_curr)]

    days = [os.path.split(dir_curr)[1][:6] for dir_curr in interval_paths]
    days = list(set(days))
    print (days)
    
    for day in days:
        path_list = []
        for view in range(4):
            path_list += glob.glob(os.path.join(data_path,'*',day+'*',str(view),'*.jpg'))
        
        print (day, len(path_list))
        # path_list = path_list[::10]
        per_cam_lists = [[] for i in range(NUM_CAMERAS)]
            
        for idx, path in enumerate(path_list):
            camera = int(get_camera_from_path(path))
            cam_idx = camera-1 
            per_cam_lists[cam_idx].append(path)

        for i in range(NUM_CAMERAS):
            print ('cam', i, len(per_cam_lists[i]))
            cam_list = per_cam_lists[i]
            with tqdm(total = len(cam_list)) as pbar:
                ims = []
                for path in cam_list:
                    pbar.update(1)
                    # print (path)
                    ims.append(imread(path))
                # if len(per_cam_lists[i])==0:
                #     continue
                ar = np.asarray(ims)
                med = np.median(ar, axis=0)
                out_file = os.path.join(out_folder,day+'_0.2fps_camera_{}.jpg'.format(i))
                imsave(out_file, med.astype('uint8'))
            print ('saved',out_file)

    visualize.writeHTMLForFolder(out_folder)


    
if __name__=='__main__':
    main()