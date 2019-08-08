import os
from helpers import util, visualize
from helpers import parse_motion_file as pms
import cv2
import numpy as np
import glob
from datetime import datetime
import pandas as pd
import subprocess
import test as dl
import sklearn
import sklearn.preprocessing
import sklearn.cluster

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


def get_contour_from_mask(mask_file, area_thresh = None, max_contours = None):

    img = cv2.imread(mask_file,cv2.IMREAD_COLOR)
    ret,thresh = cv2.threshold(img,127,255,0)
    contours, hierarchy = cv2.findContours(thresh[:,:,0], cv2.RETR_EXTERNAL, 2)
    
    if len(contours)<1:
        return None

    if max_contours is not None and len(contours)>max_contours:
        return None

    contours = contours[0]
    im_area= img.shape[0]*img.shape[1]
    if area_thresh is not None:
        cont_area = cv2.contourArea(contours)
        percent = cont_area/float(im_area)
        if percent<area_thresh:
            return None
    return contours

def cluster_contours(X,num_clusters):
    
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    kmeans = sklearn.cluster.KMeans(num_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels

def save_contour_clusters(contours, out_file, num_clusters):
    contour_length = [len(contour) for contour in contours]
    max_dim = np.max(contour_length)*2
    print (max_dim/2)
    contours_padded = []
    for contour in contours:
        contour = np.array(contour).flatten()
        contour = np.concatenate([contour,np.zeros(max_dim - contour.size)])
        contours_padded.append(contour[np.newaxis,:])
    contours_padded = np.concatenate(contours_padded,axis = 0)
    cluster_labels = cluster_contours(contours_padded,num_clusters)
    labels = np.unique(cluster_labels)
    np.save(out_file,cluster_labels)

def script_extract_motion_frames():
    in_dir = '../data/tester_kit/naughty_but_nice'
    motion_files = glob.glob(os.path.join(in_dir,'*.txt'))
    motion_files.sort()

    # print (motion_files)
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

        cam, vid_start_time = pms.get_vid_info(video_file)

        df = pms.read_motion_file(motion_file)
        motion_str = 'Motion Detection Started'
        rows = df.loc[(df['Minor Type']==motion_str),['Date Time']]
        
        motion_times =rows.iloc[:,0].values
        vid_times = motion_times - vid_start_time
        vid_times = vid_times.astype(np.datetime64)
        
        commands= [extract_frame_at_time(video_file, out_dir, time_curr) for time_curr in vid_times]
        commands = set(commands)
        commands_file =os.path.join(out_dir,'commands.txt') 
        # print (commands[0])        
        # print (commands_file)
        util.writeFile(commands_file, commands)
        # break
        for command in commands:
            subprocess.call(command, shell=True)
            
        visualize.writeHTMLForFolder(out_dir, height=256, width = 448)


def script_run_deeplab_on_motion_frames():
    out_dir_meta = os.path.join('../scratch','frames_at_motion')
    util.mkdir(out_dir_meta)

    vid_names = ['ch01_20181209092600'  ,'ch03_20181209092844'  ,'ch05_20181208230458'  ,'ch06_20181209093714']
    vid_dirs = [os.path.join(out_dir_meta, vid_name) for vid_name in vid_names]

    model_path = '../data/deeplab_xception_coco_voctrainval.tar.gz'
    model = dl.DeepLabModel(model_path)

    for vid_dir in vid_dirs:
        im_files = glob.glob(os.path.join(vid_dir,'*.jpg'))
        im_files.sort()
        out_dir = os.path.join(vid_dir,'seg_maps')
        util.mkdir(out_dir)
        for im_file in im_files:
            out_file = os.path.join(out_dir,os.path.split(im_file)[1].replace('.jpg','.png'))
            dl.run_visualization(model,im_file, out_file)
        visualize.writeHTMLForFolder(out_dir,ext = '.png',height = 256, width = 448)


def script_getting_contours():
    out_dir_meta = os.path.join('../scratch','frames_at_motion')
    util.mkdir(out_dir_meta)
    vid_names = ['ch01_20181209092600'  ,'ch03_20181209092844'  ,'ch05_20181208230458'  ,'ch06_20181209093714']
    vid_dirs = [os.path.join(out_dir_meta, vid_name) for vid_name in vid_names]
    for vid_dir in vid_dirs:
        seg_dir = os.path.join(vid_dir,'seg_maps')
        mask_files = glob.glob(os.path.join(seg_dir,'*.png'))
        files = []
        contours = []
        for mask_file in mask_files:
            contour = get_contour_from_mask(mask_file, area_thresh = 0.05, max_contours = 1)
            if contour is not None:
                contours.append(contour)
                files.append(mask_file)
        out_file_contours = os.path.join(vid_dir,'contours.npz')
        files = np.array(files)
        contours = np.array(contours)
        print (vid_dir, out_file_contours)
        print (len(mask_files), len(files), len(contours))
        np.savez(out_file_contours, files = files, contours = contours)

def script_cluster_contours(num_clusters):
    out_dir_meta = os.path.join('../scratch','frames_at_motion')
    util.mkdir(out_dir_meta)
    vid_names = ['ch01_20181209092600'  ,'ch03_20181209092844'  ,'ch05_20181208230458'  ,'ch06_20181209093714']
    vid_dirs = [os.path.join(out_dir_meta, vid_name) for vid_name in vid_names]

    # num_clusters = 50
    # max_dim = 487
    contours_all = []
    files_all = []
    for vid_dir in vid_dirs:
        contour_file = os.path.join(vid_dir, 'contours.npz')
        data = np.load(contour_file, allow_pickle= True)
        files = data['files']
        contours = data['contours']
        contours_all+=list(contours)
        files_all+=list(files)
        # out_file = os.path.join(vid_dir, 'labels_self_'+str(num_clusters)+'.npy')
        # save_contour_clusters(contours, out_file, num_clusters)
    vid_dir = os.path.join(out_dir_meta, 'all_vids')
    util.mkdir(vid_dir)
    out_file_contours = os.path.join(vid_dir,'contours.npz')
    files = np.array(files_all)
    contours = np.array(contours_all)
    np.savez(out_file_contours, files = files, contours = contours)
    out_file = os.path.join(vid_dir, 'labels_self_'+str(num_clusters)+'.npy')
    save_contour_clusters(contours, out_file, num_clusters)




def main():
    num_clusters = 50
    # script_cluster_contours(num_clusters)
    # return
    out_dir_meta = os.path.join('../scratch','frames_at_motion')
    util.mkdir(out_dir_meta)
    # vid_names = ['ch01_20181209092600'  ,'ch03_20181209092844'  ,'ch05_20181208230458'  ,'ch06_20181209093714']
    vid_names = ['all_vids']
    vid_dirs = [os.path.join(out_dir_meta, vid_name) for vid_name in vid_names]

    # max_dim = 487
    # dir_server = '/home/maheen/gross_pain'
    str_replace = ['..','']
    
    for vid_dir in vid_dirs:
        contour_file = os.path.join(vid_dir, 'contours.npz')
        data = np.load(contour_file, allow_pickle= True)
        files = data['files']
        contours = data['contours']
        labels_file = os.path.join(vid_dir, 'labels_self_'+str(num_clusters)+'.npy')
        labels = np.load(labels_file)

        out_file_html = os.path.join(vid_dir, 'cluster_viz_'+str(num_clusters)+'.html')

        im_paths = []
        captions = []
        for label in np.unique(labels):
            im_paths_curr = files[labels==label]
            im_paths_curr = [im_path.replace(str_replace[0], str_replace[1]) for im_path in im_paths_curr]
            captions_curr = ['/'.join(im_path.split('/')[-3::2]) for im_path in im_paths_curr]
            im_paths.append(im_paths_curr)
            captions.append(captions_curr)

        visualize.writeHTML(out_file_html, im_paths, captions, height = 256, width = 488)
        print (out_file_html)
        # input()








        

if __name__=='__main__':
    main()
