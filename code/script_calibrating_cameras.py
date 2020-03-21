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


def extract_calibration_vid_frames():
    data_path = '../data/camera_calibration_videos'
    out_dir = '../data/camera_calibration_frames_redo'
    data_selection_path = '../metadata/interval_calibration.csv'
    util.mkdir(out_dir)

    mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1., output_dir = out_dir,
                 views = [0,1,2,3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
    mve.extract_frames()

    # mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1/5., output_dir = out_dir,
 #                 views = [3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
    # mve.extract_frames(replace = True, subjects_to_extract=['cell2'])
    
    dirs_to_check = [os.path.join(out_dir, 'cell1','20200317130703_131800'), os.path.join(out_dir,'cell2','20200317130703_131800')]
    view_multiple_dirs(dirs_to_check, out_dir)

def fix_filenames():
    vid_dir = '../data/camera_calibration_videos/original_videos/2020-03-17'
    out_dir = '../data/camera_calibration_videos/cell1/2020-03-17'
    util.makedirs(out_dir)

    # get video names
    vid_files = glob.glob(os.path.join(vid_dir,'*.mp4'))
    vid_files.sort()

    # # extract first frame
    # for vid_file in vid_files:
    #   # command = ['ffmpeg', '-i', vid_file, '-vf', '"select=eq(n\,0)"', '-q:v', '3', vid_file[:vid_file.rindex('.')]+'.jpg']
    #   command = ['ffmpeg', '-i', vid_file, '-y', '-vframes', '1', '-f', 'image2', vid_file[:vid_file.rindex('.')]+'.jpg']
    #   subprocess.call(command)

    # # view first frames
    # visualize.writeHTMLForFolder(vid_dir, height = 506, width = 896)

    # # what're the times i see
    new_times = ['130659', '130657', '130659', '130659', '131240', '130659', '130659','131150', '130659', '130659','131010']
    
    # rename files
    for vid_file, new_time in zip(vid_files, new_times):
        vid_name = os.path.split(vid_file)[1]
        vid_name = vid_name[:13]+new_time+'.mp4'
        out_file = os.path.join(out_dir, vid_name)
        command = ['cp',vid_file, out_file]
        print (' '.join(command))
        subprocess.call(command)

#  goes through a list of images and returns the ones with visible checkboard
def get_img_chessboard(fname):
    check_cols = 7
    check_rows = 9
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect chessboard
    ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows),cv2.CALIB_CB_FAST_CHECK)
    
    if ret == True:
        return fname
        

def do_intrinsic(im_list, out_dir = False):
    check_cols = 7
    check_rows = 9
    h = 1520
    w = 2688

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((check_cols*check_rows,3), np.float32)
    objp[:,:2] = np.mgrid[0:check_cols,0:check_rows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # load list of images
    for idx_fname, fname in enumerate(im_list):
        img = cv2.imread(fname)
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # detect chessboard
        ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows),None)
        
        if ret == True:
            objpoints.append(objp)

            # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # imgpoints.append(corners2)
            imgpoints.append(corners)

            if out_dir:
                out_file = os.path.join(out_dir,'check_%02d.jpg'%idx_fname)
                # Draw and display the corners
                dst = cv2.drawChessboardCorners(img, (check_cols,check_rows), corners,ret)
                cv2.imwrite(out_file,dst)

    # calibrate the camera
    # mtx = cv2.initCameraMatrix2D(objpoints, imgpoints, (h,w))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h,w), None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    
    if out_dir:
        for idx_fname, fname in enumerate(im_list):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            out_file = os.path.join(out_dir,'undistort_%02d.jpg'%idx_fname)
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            
            cv2.imwrite(out_file,dst)


def main():
    # fix_filenames()
    # extract_calibration_vid_frames()

    meta_dir = '../data/camera_calibration_frames_redo'
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    util.mkdir(out_dir)
    
    # get im with chessboard found
    for idx_cell,cell_str in enumerate(['cell1', 'cell2']):
        for view in range(4):
            
            dir_curr = os.path.join(meta_dir,cell_str,interval_str,str(view))
            im_list = glob.glob(os.path.join(dir_curr,'*.jpg'))
            im_list.sort()
            if idx_cell==0:
                im_list = im_list[:len(im_list)//2]
            else:
                # no one's in cell 2 till second half of videos
                im_list = im_list[len(im_list)//2:]
            print (len(im_list))
            
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            ret_vals = pool.map(get_img_chessboard,im_list)
            ret_vals = [val for val in ret_vals if val is not None]
            print (len(ret_vals))
            
            out_file = os.path.join(out_dir,'_'.join([cell_str,str(view)])+'.txt')
            util.writeFile(out_file, ret_vals)



    return
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_meta = '../scratch/viewing_intrinsic_calib'
    util.mkdir(out_dir_meta)

    interval_str = '20200317130703_131800'
    frame_file = '../metadata/calibration_frames.csv'
    frame_df = pd.read_csv(frame_file)
    print (frame_df)

    for idx_row,row in frame_df.iterrows():
        im_fmt = '_'.join(['ce','00',str(row['view']),'%06d.jpg'])
        im_dir = os.path.join(meta_dir,'cell'+str(row['cell']),interval_str,str(row['view']))
        im_list = [os.path.join(im_dir,im_fmt%im_idx) for im_idx in range(row['start_frame'],row['end_frame']+1)]
        out_dir = os.path.join(out_dir_meta,'cam_'+str(row['cam']))
        util.mkdir(out_dir)
        do_intrinsic(im_list, out_dir)
        visualize.writeHTMLForFolder(out_dir,height = 506, width = 896)
        print (out_dir)
        
    




    


    # out_dir_html = '../data/camera_calibration_frames'
    # dirs_to_check = [os.path.join(out_dir_html, 'cell1','20200317130703_131800'), os.path.join(out_dir_html,'cell2','20200317130703_131800')]
    # view_multiple_dirs(dirs_to_check, out_dir_html)



    
    


if __name__=='__main__':
    main()
