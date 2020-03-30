import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread, imsave

NUM_CAMERAS = 8
frame_folder = '../data/intervals_for_extraction_128_128_0.1fps/'


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
    

path_list = get_all_jpg_paths(frame_folder)

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
        imsave('median_0.1fps_camera_{}.jpg'.format(i), med.astype('uint8'))
    
