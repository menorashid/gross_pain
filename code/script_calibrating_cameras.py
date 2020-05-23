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
import itertools
import matplotlib.pyplot as plt
import shutil

check_rows = 11
# 11
check_cols = 7
h = 1520
w = 2688

def extract_calibration_vid_frames():
    data_path = '../data/camera_calibration_videos_try2'
    out_dir = '../data/camera_calibration_frames_try2'
    data_selection_path = '../metadata/interval_calibration_try2.csv'
    util.mkdir(out_dir)

    mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1., output_dir = out_dir,
                 views = [0,1,2,3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
    mve.extract_frames()

    # mve = MultiViewFrameExtractor(data_path = data_path, width= 2688, height = 1520, frame_rate = 1/5., output_dir = out_dir,
 #                 views = [3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count())
    # mve.extract_frames(replace = True, subjects_to_extract=['cell2'])
    
    # dirs_to_check = [os.path.join(out_dir, 'cell1','20200317130703_131800'), os.path.join(out_dir,'cell2','20200317130703_131800')]
    view_multiple_dirs(dirs_to_check, out_dir)

def fix_filenames():
    out_dir = '../data/camera_calibration_videos/cell1/2020-03-17'
    vid_dir = '../data/camera_calibration_videos/original_videos/2020-03-17'
    
    out_dir = '../data/camera_calibration_videos_try2/cell1/2020-04-28'
    vid_dir = '../data/camera_calibration_videos_try2/original_videos/2020-04-28'
    
    util.makedirs(out_dir)

    # get video names
    vid_files = glob.glob(os.path.join(vid_dir,'*.mp4'))
    vid_files.sort()

    # extract first frame
    # for vid_file in vid_files:
    #   # command = ['ffmpeg', '-i', vid_file, '-vf', '"select=eq(n\,0)"', '-q:v', '3', vid_file[:vid_file.rindex('.')]+'.jpg']
    #   command = ['ffmpeg', '-i', vid_file, '-y', '-vframes', '1', '-f', 'image2', vid_file[:vid_file.rindex('.')]+'.jpg']
    #   subprocess.call(command)

    # # view first frames
    # visualize.writeHTMLForFolder(vid_dir, height = 506, width = 896)

    # # # what're the times i see
    # for vid_file in vid_files:
    #     print (vid_file)
    # new_times = ['130659', '130657', '130659', '130659', '131240', '130659', '130659','131150', '130659', '130659','131010']
    new_times = ['104446','105105','111923',
                '104441','110408',
                '104446','105728','112545',
                '104446','111210',
                '104446','110735','113552',
                '104446','110440','113255',
                '104446','110203','113020',
                '104446','105052','111908']
    
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
    global check_rows, check_cols
    # print (check_rows, check_cols)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # detect chessboard
    ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows),cv2.CALIB_CB_FAST_CHECK)
    
    if ret == True:
        # out_file = '../scratch/check_cell2.jpg'
        # dst = cv2.drawChessboardCorners(img, (check_cols,check_rows), corners,ret)
        # cv2.imwrite(out_file,dst)
        return fname
        

def do_intrinsic(im_list, out_dir = False):
    global check_rows, check_cols, h, w
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((check_cols*check_rows,3), np.float32)
    objp[:,:2] = np.mgrid[0:check_cols,0:check_rows].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # load list of images
    for idx_fname, fname in enumerate(im_list):
        print (idx_fname)
        img = cv2.imread(fname)
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # detect chessboard
        ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows),None)
        assert ret 
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
    mtx = cv2.initCameraMatrix2D(objpoints, imgpoints, (w,h))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), mtx, None, flags = cv2.CALIB_USE_INTRINSIC_GUESS)
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

    return mtx, dist


def get_all_files_with_chessboard():

    meta_dir = '../data/camera_calibration_frames_redo'
    interval_str = '20200317130703_131800'

    meta_dir = '../data/camera_calibration_frames_try2'
    interval_str = '20200428104445_113700'

    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    util.mkdir(out_dir)
    
    # get im with chessboard found
    for idx_cell,cell_str in enumerate(['cell1', 'cell2']):
        for view in range(4):
            
            dir_curr = os.path.join(meta_dir,cell_str,interval_str,str(view))
            im_list = glob.glob(os.path.join(dir_curr,'*.jpg'))
            im_list.sort()
            lim = 1650
            if idx_cell==0:
                im_list = im_list[:lim]
            else:
                # no one's in cell 2 till second half of videos
                im_list = im_list[lim:]
            print (len(im_list))
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            ret_vals = pool.map(get_img_chessboard,im_list)
            ret_vals = [val for val in ret_vals if val is not None]
            print (len(ret_vals))
            
            out_file = os.path.join(out_dir,'_'.join([cell_str,str(view)])+'.txt')
            util.writeFile(out_file, ret_vals)

def get_common_im(out_dir, cell_strs = ['cell1','cell2'], views = [0,1,2,3]):
    common_im_dict = {}
    info_arr = []
    im_arr = []
    for cell_str,view in itertools.product(cell_strs, views):            
        curr_file = os.path.join(out_dir,'_'.join([cell_str,str(view)+'.txt']))
        for im_file in util.readLinesFromFile(curr_file):
            im_num = os.path.split(im_file)[1].split('_')[-1]
            im_num = int(im_num[:im_num.rindex('.')])
            info_row = [int(cell_str[-1]),view,im_num]
            info_arr.append(info_row)
            im_arr.append(im_file)

    info_arr = np.array(info_arr)
    im_arr = np.array(im_arr)
    
    for cell_num in [int(cell_str[-1]) for cell_str in cell_strs]:
        combos = [itertools.combinations(views,num_views) for num_views in range(1,len(views)+1)]
        for combo in combos:
            for views in combo:
                views = list(views)
                key_curr = tuple([cell_num]+views)
                kept_im = np.unique(info_arr[:,-1])
                bin_views = []
                for view in views:
                    bin_im = np.logical_and(info_arr[:,0]==cell_num,info_arr[:,1]==view)
                    bin_im_keep = np.in1d(kept_im, info_arr[bin_im,-1])
                    kept_im = kept_im[bin_im_keep]
                    bin_views.append(bin_im)

                bin_kept = np.in1d(info_arr[:,-1],kept_im)
                im_cols = []
                for bin_im in bin_views:
                    bin_im = np.logical_and(bin_im, bin_kept)
                    im_col = im_arr[bin_im]
                    im_cols.append(list(im_col))

                common_im_dict[key_curr] = im_cols

    return common_im_dict

def rough_script_intrinsic():
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


def visualize_all_files_with_chessboard():
    meta_dir = '../data/camera_calibration_frames_redo'
    interval_str = '20200317130703_131800'

    meta_dir = '../data/camera_calibration_frames_try2'
    interval_str = '20200428104445_113700'

    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    

    out_dir_html = os.path.join(meta_dir,'common_im_html')
    str_replace = ['..','/gross_pain']
    util.mkdir(out_dir_html)

    common_im_dict = get_common_im(out_dir,['cell2'])
        # ,1,2,3])
    for key_curr in common_im_dict.keys():

        im_cols = common_im_dict[key_curr]
        im_cols = [im_col[::5] for im_col in im_cols]
        print (key_curr, len(im_cols),len(im_cols[0]))
        # im_cols = np.array(im_cols)
        write_common_html(im_cols, out_dir_html, str_replace, key_curr)
        # if len(im_cols[0])>0:
        #     out_file_html = os.path.join(out_dir_html,'_'.join([str(val) for val in key_curr])+'.html')
        #     captions =[]
        #     for idx_im_col in range(len(im_cols)):
        #         captions.append([])
        #         for idx_im in range(len(im_cols[idx_im_col])):
        #             # print (im_cols[idx_im_col][idx_im])
        #             # s = input()
        #             im_cols[idx_im_col][idx_im] = im_cols[idx_im_col][idx_im].replace(str_replace[0],str_replace[1])
        #             captions[idx_im_col].append(os.path.split(im_cols[idx_im_col][idx_im])[1])
        #     views = list(key_curr)[1:]
        #     # captions = [[str(view)]*len(im_cols[idx_view]) for idx_view, view in enumerate(views)]
        #     visualize.writeHTML(out_file_html, im_cols, captions)

def write_common_html(im_cols, out_dir_html, str_replace, key_curr):
    if len(im_cols[0])>0:
        out_file_html = os.path.join(out_dir_html,'_'.join([str(val) for val in key_curr])+'.html')
        captions =[]
        for idx_im_col in range(len(im_cols)):
            captions.append([])
            for idx_im in range(len(im_cols[idx_im_col])):
                # print (im_cols[idx_im_col][idx_im])
                # s = input()
                im_cols[idx_im_col][idx_im] = im_cols[idx_im_col][idx_im].replace(str_replace[0],str_replace[1])
                captions[idx_im_col].append(os.path.split(im_cols[idx_im_col][idx_im])[1])
        # views = list(key_curr)[1:]
        # captions = [[str(view)]*len(im_cols[idx_view]) for idx_view, view in enumerate(views)]
        visualize.writeHTML(out_file_html, im_cols, captions)


def save_common_im_with_chessboard_det():
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_chess_det = '../data/camera_calibration_frames_withChessboardDet'
    interval_str = '20200317130703_131800'
    str_replace = [meta_dir,'/gross_pain/data/camera_calibration_frames_withChessboardDet']
    
    meta_dir = '../data/camera_calibration_frames_try2'
    out_dir_chess_det = os.path.join(meta_dir, 'common_im_withChessboardDet')
    interval_str = '20200428104445_113700'
    str_replace = [meta_dir,out_dir_chess_det.replace('..','/gross_pain')]

    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_html = os.path.join(meta_dir,'common_im_withChessboardDet_html')
    util.mkdir(out_dir_chess_det)
    util.mkdir(out_dir_html)

    global check_cols, check_rows

    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict.keys():
        # print (key_curr, len(key_curr))
        if len(key_curr)<4:
            continue
        
        im_cols = common_im_dict[key_curr]
        for idx_im_col, im_col in enumerate(im_cols):
            print (len(im_col))
            for idx_im_file, im_file in enumerate(im_col):
                print (key_curr, idx_im_col, idx_im_file, len(im_col))
                out_file = im_file.replace(meta_dir, out_dir_chess_det)
                # if os.path.exists(out_file):
                #     continue

                util.makedirs(os.path.split(out_file)[0])
                

                # check_cols = 7
                # check_rows = 9
                img = cv2.imread(im_file)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # detect chessboard
                ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows),cv2.CALIB_CB_FAST_CHECK)
                # print (corners.shape)
                np.save(out_file.replace('.jpg','.npy'),corners)
                # assert ret
                dst = cv2.drawChessboardCorners(img, (check_cols,check_rows), corners,ret)
                cv2.imwrite(out_file,dst)
                # s = input()

        write_common_html(im_cols, out_dir_html, str_replace, key_curr)
        # if len(im_cols[0])>0:
        #     out_file_html = os.path.join(out_dir_html,'_'.join([str(val) for val in key_curr])+'.html')
        #     for idx_im_col in range(len(im_cols)):
        #         for idx_im in range(len(im_cols[idx_im_col])):
        #             im_cols[idx_im_col][idx_im] = im_cols[idx_im_col][idx_im].replace(str_replace[0],str_replace[1])
        #     views = list(key_curr)[1:]
        #     captions = [[' '.join([str(views[idx_im_col]),os.path.split(im_file)[1][-10:-4]]) for im_file in im_col] for idx_im_col,im_col in enumerate(im_cols)]
        #     # captions = [[str(view)]*len(im_cols[idx_view]) for idx_view, view in enumerate(views)]
        #     visualize.writeHTML(out_file_html, im_cols, captions)

def script_save_all_intrinsics():
    meta_dir = '../data/camera_calibration_frames_redo'
    interval_str = '20200317130703_131800'

    meta_dir = '../data/camera_calibration_frames_try2'
    interval_str = '20200428104445_113700'

    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_intrinsic = os.path.join(meta_dir,'intrinsics')
    util.mkdir(out_dir_intrinsic)

    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict:
        if len(key_curr)>2:
            continue

        im_files = common_im_dict[key_curr]
        assert len(im_files)==1
        im_files = im_files[0][::2] #using every second because of too many images

        mtx, dist = do_intrinsic(im_files)
        
        print (mtx, type(mtx))
        print (dist, type(dist))
        out_file = os.path.join(out_dir_intrinsic,'_'.join([str(val) for val in key_curr])+'.npz')
        print (out_file)
        np.savez(out_file, mtx = mtx, dist = dist)
        # s = input()

def save_im_chessboard_dets(arg):
    (im_file, out_file, out_file_viz) = arg
    num_dets = 4 
    # check_cols = 7 
    # check_rows = 9
    try:
        print (im_file)
        img = cv2.imread(im_file)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners_list = []
        for det_num in range(num_dets):
            ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows),None)
            
            corners = corners.squeeze()
            corners_list.append(corners)

            min_vals = np.min(corners, axis= 0).astype(int)

            max_vals = np.max(corners, axis = 0).astype(int)
            gray[min_vals[1]:max_vals[1],min_vals[0]:max_vals[0]] = 0
            print(out_file_viz)
            cv2.imwrite(out_file_viz, gray)
            s = input()

        corners_mean = np.array([np.mean(vals, axis = 0) for vals in corners_list])
        order = []
        for to_inc in [[0,1],[2,3]]:
            y_ord = np.argsort(corners_mean[:,1])[to_inc]
            top_row = corners_mean[y_ord,:]
            x_ord = np.argsort(top_row[:,0])
            order += [y_ord[x_ord[0]],y_ord[x_ord[1]]]
        
        corners_list = np.array(corners_list)
        corners_list = corners_list[order,:,:]
        np.save(out_file, corners_list)

        if out_file_viz is not None:
            colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
            for idx_idx in range(corners_list.shape[0]):
                color_curr = colors[idx_idx]
                corners = corners_list[idx_idx]
                for corner in corners:
                    cv2.drawMarker(img, (corner[0],corner[1]),color_curr, markerType=cv2.MARKER_STAR, markerSize=4, thickness=2)
            cv2.imwrite(out_file_viz, img)

        return True
    except:
        return False


def save_and_order_all_chessboard_dets():
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_chess_det = '../data/camera_calibration_frames_withChessboardDet'
    util.mkdir(out_dir_chess_det)
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_dets = os.path.join(meta_dir, 'chessboard_dets')
    out_dir_viz = os.path.join(meta_dir, 'chessboard_dets_viz')
    util.mkdir(out_dir_dets)
    util.mkdir(out_dir_viz)

    out_dir_scratch = os.path.join('../scratch/chessboard_dets')
    util.mkdir(out_dir_scratch)

    im_files = []
    for file_curr in glob.glob(os.path.join(out_dir,'*.txt')):
        im_files+=util.readLinesFromFile(file_curr)
    assert len(im_files)==len(list(set(im_files)))
    print (len(im_files))

    args = []

    for idx_im_file, im_file in enumerate(im_files):
        out_file_corners = im_file.replace(meta_dir,out_dir_dets)[:-4]+'.npy'
        util.makedirs(os.path.split(out_file_corners)[0])

        out_file_viz = im_file.replace(meta_dir, out_dir_viz)
        util.makedirs(os.path.split(out_file_viz)[0])
        
        if os.path.exists(out_file_corners):
            continue

        args.append((im_file, out_file_corners, out_file_viz))

    # print (len(im_files),len(args))
    # for arg in args:
    #     save_im_chessboard_dets(arg)
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    ret_vals = pool.map(save_im_chessboard_dets,args)
    ret_vals = np.array(ret_vals)
    print (ret_vals.shape, np.sum(ret_vals), np.sum(ret_vals==0))
    out_file = os.path.join(out_dir_dets,'problem_cases.npy')
    np.save(out_file, ret_vals)

    ret_vals = np.load(out_file)
    print (ret_vals.shape, np.sum(ret_vals==0))


def save_htmls_for_ordered_chessboards():
    meta_dir = '../data/camera_calibration_frames_redo'
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_dets = os.path.join(meta_dir, 'chessboard_dets')
    out_dir_viz = os.path.join(meta_dir, 'chessboard_dets_viz')

    out_dir_html = os.path.join(meta_dir,'common_im_withChessboardDet_ordered_html')
    str_replace = [meta_dir,'/gross_pain/data/camera_calibration_frames_redo/chessboard_dets_viz']
    util.mkdir(out_dir_html)

    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict.keys():
        im_cols = common_im_dict[key_curr]
        if len(im_cols[0])>0:
            out_file_html = os.path.join(out_dir_html,'_'.join([str(val) for val in key_curr])+'.html')
            for idx_im_col in range(len(im_cols)):
                for idx_im in range(len(im_cols[idx_im_col])):
                    im_cols[idx_im_col][idx_im] = im_cols[idx_im_col][idx_im].replace(str_replace[0],str_replace[1])
            views = list(key_curr)[1:]
            captions = [[' '.join([str(views[idx_im_col]),os.path.split(im_file)[1][-10:-4]]) for im_file in im_col] for idx_im_col,im_col in enumerate(im_cols)]
            # captions = [[str(view)]*len(im_cols[idx_view]) for idx_view, view in enumerate(views)]
            visualize.writeHTML(out_file_html, im_cols, captions)

import math
def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)

    # roll, pitch, yaw
    return psi*180/math.pi, theta*180/math.pi, phi*180/math.pi


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def script_stereo_calibrate():
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_chess_det = '../data/camera_calibration_frames_withChessboardDet'
    util.mkdir(out_dir_chess_det)
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_intrinsic = os.path.join(meta_dir, 'intrinsics')
    out_dir_dets = os.path.join(meta_dir, 'chessboard_dets')
    # check_cols = 7
    # check_rows = 9
    # h = 1520
    # w = 2688

    cell_num = '1'
    views = ['0','3']
    im_num = '000152'
    dets_all = []
    det_path_meta = os.path.join(out_dir_dets,'cell'+cell_num,interval_str)
    det_files_all = []
    for view in views:
        det_files = np.array([os.path.split(file_curr)[1][-10:-4] for file_curr in glob.glob(os.path.join(det_path_meta,view,'ce_00_'+view+'_*.npy'))])
        det_files_all.append(det_files)

    bin_keep = np.in1d(det_files_all[0], det_files_all[1])
    im_nums = det_files_all[0][bin_keep]
    im_num = im_nums[-1]
    im_nums = [im_num]
    im_files = [os.path.join(meta_dir,'cell'+cell_num,interval_str,view_curr,'ce_00_'+view_curr+'_'+im_num+'.jpg') for view_curr in views]

    dets_all = []
    for idx_view, view in enumerate(views):
        dets_curr = []
        for im_num in im_nums[:1]:
            dets_path = os.path.join(det_path_meta, view,'ce_00_'+view+'_'+im_num+'.npy')
            dets = np.load(dets_path)[:1]
            # assert dets.shape[0]==4
            dets_curr.append(dets)
        dets_all.append(np.concatenate(dets_curr,axis = 0))

    # intrinsics 
    mtxs = []
    dists = []
    for view in views:
        file_int = os.path.join(out_dir_intrinsic,cell_num+'_'+view+'.npz')
        vals = np.load(file_int)
        # print (vals['mtx'])
        # print (vals['dist'])
        mtxs.append(vals['mtx'])
        dists.append(vals['dist'])

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((check_cols*check_rows,3), np.float32)
    objp[:,:2] = np.mgrid[0:check_cols,0:check_rows].T.reshape(-1,2)
    objp = objp[np.newaxis,:,:]
    objp = np.tile(objp,(dets_all[0].shape[0],1,1))
    print (objp.shape)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objp,dets_all[0],dets_all[1],  mtxs[0],dists[0],mtxs[1],dists[1],(w,h),flags = cv2.CALIB_FIX_INTRINSIC)
    print (R)
    print (euler_angles_from_rotation_matrix(R))
    print (T)
    print (retval)
    print (dists[0],distCoeffs1)
    
    print ('E',E/E[2,2])
    print ('F',F)

    E_us = np.matmul(cameraMatrix2.T,np.matmul(F,cameraMatrix1))
    print ('E_us',E_us/E_us[2,2])

    R1_us, R2_us, t_us =   cv2.decomposeEssentialMat(E)
    print ('R',R)
    print (T.shape)
    print ('T',T/T[2])
    print ('R1_us',R1_us)
    print ('R2_us',R2_us)
    print ('t_us', t_us/t_us[2])

    return




    frames = [cv2.imread(im_file) for im_file in im_files]
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(dets_all[1].squeeze(), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(frames[0],frames[1],lines1,dets_all[0].squeeze(),dets_all[1].squeeze())
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(dets_all[0].squeeze(), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(frames[1],frames[0],lines2,dets_all[1].squeeze(),dets_all[0].squeeze())




    # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w,h), R, T, cv2.CALIB_ZERO_DISPARITY)

    # print (R1,R2)

    # mapxL, mapyL = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w,h), cv2.CV_32FC1)
    # mapxR, mapyR = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w,h), cv2.CV_32FC1)

    
    # dstL = cv2.remap(frames[0], mapxL, mapyL,cv2.INTER_LINEAR)
    # dstR = cv2.remap(frames[1], mapxR, mapyR,cv2.INTER_LINEAR)
    out_file = '../scratch/0.jpg'
    cv2.imwrite(out_file, img5)
    out_file = '../scratch/1.jpg'
    cv2.imwrite(out_file, img3)
    out_file = '../scratch/0_org.jpg'
    cv2.imwrite(out_file, frames[0])
    out_file = '../scratch/1_org.jpg'
    cv2.imwrite(out_file, frames[1])
    visualize.writeHTMLForFolder('../scratch')


def script_calibrate_manual():
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_intrinsic = os.path.join(meta_dir, 'intrinsics')

    out_dir = os.path.join(meta_dir, 'manual_corr_int_ext')
    util.mkdir(out_dir)
    dir_curr = '../data/camera_calibration_frames_redo/to_copy_local_manual'
    viz = True


    cell_num = '1'
    views = ['0','3']  
    views_sort = views[:]
    views_sort.sort()
    dir_curr = os.path.join(dir_curr, '_'.join([cell_num]+views_sort))
    if viz:
        out_dir_viz = os.path.join(out_dir,'_'.join([cell_num]+views))
        util.mkdir(out_dir_viz)


    pt_files = list(glob.glob(os.path.join(dir_curr,views[0],'*.npy')))
    pt_files.sort()
    print(pt_files)
    pts_all =[]
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in range(8)]

    for idx_pt_file, pt_file in enumerate(pt_files):
        pts = np.load(pt_file)
        
        if viz:
            im_files = [pt_file.replace('.npy','.jpg'),pt_file.replace('/'+views[0]+'/','/'+views[1]+'/').replace('.npy','.jpg')]
            for idx_im,im_file in enumerate(im_files):
                out_file = os.path.join(out_dir_viz,str(idx_pt_file)+'_'+str(idx_im)+'.jpg')
                im = cv2.imread(im_file)
                for idx_pt, pt in enumerate(pts[idx_im]):
                    im = cv2.circle(im,tuple(np.int32(pt)),5,colors[idx_pt],-1)
                cv2.imwrite(out_file, im)

        print (pts.shape)
        pts_all.append(pts)



    pts_all = np.concatenate(pts_all,axis = 1)
    pts_all =[pts_all[0], pts_all[1]]

    F, mask = cv2.findFundamentalMat(pts_all[0],pts_all[1],cv2.FM_RANSAC,5,0.9999)

    # get intrinsics
    mtxs = []
    dists = []
    for view in views:
        file_int = os.path.join(out_dir_intrinsic,cell_num+'_'+view+'.npz')
        vals = np.load(file_int)
        # print (vals['mtx'])
        # print (vals['dist'])
        mtxs.append(vals['mtx'])
        dists.append(vals['dist'])

    E = np.matmul(mtxs[1].T,np.matmul(F,mtxs[0]))
    print ('E',E/E[2,2])

    R1, R2, T = cv2.decomposeEssentialMat(E)
    print (R1)
    print (euler_angles_from_rotation_matrix(R1))

    print (R2)
    print (euler_angles_from_rotation_matrix(R2))    

    print (T)
    # R1

    print (F)
    # We select only inlier points
    print (mask.shape, pts_all[0].shape)
    # pts_all[0] = pts_all[0][mask.ravel()==1]
    # pts_all[1] = pts_all[1][mask.ravel()==1]
    out_file = os.path.join(out_dir,'_'.join([cell_num]+views)+'.npz')
    np.savez(out_file,R1 = R1, R2=R2, T = T, K1 = mtxs[0], K2 = mtxs[1], d1 = dists[0], d2 = dists[1])

    if viz:
        im_files = [os.path.join(dir_curr,view,os.path.split(pt_files[-1])[1][:-4]+'.jpg') for view in views]
        frames = [cv2.imread(im_file) for im_file in im_files]
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts_all[1].squeeze(), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(np.array(frames[0]),np.array(frames[1]),lines1,np.int32(pts_all[0]).squeeze(),np.int32(pts_all[1]).squeeze())
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts_all[0].squeeze(), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = drawlines(np.array(frames[1]),np.array(frames[0]),lines2,np.int32(pts_all[1]).squeeze(),np.int32(pts_all[0]).squeeze())
        out_file = os.path.join(out_dir_viz,'epilines_0.jpg')
        cv2.imwrite(out_file, img5)
        out_file = os.path.join(out_dir_viz,'epilines_1.jpg')
        cv2.imwrite(out_file, img3)
        visualize.writeHTMLForFolder(out_dir_viz)

def get_im_path(meta_dir, interval_str, cell_num, view, im_num):
    cell_num = str(cell_num)
    view = str(view)
    im_str= '%06d.jpg'%im_num
    path = os.path.join(meta_dir, 'cell'+cell_num, interval_str, view, '_'.join(['ce','00',view,im_str]))
    return path

def script_calibrate_center_board():
    # from checkerboard import detect_checkerboard
    # select an image
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_intrinsic = os.path.join(meta_dir, 'intrinsics')
    interval_str = '20200317130703_131800'
    viz = True

    # cell_num = 1
    # im_num = 271
    # views = [0,3,1,2]
    # file_dets = ['../data/camera_calibration_frames_redo/to_copy_local_manual/1_0_3/0/000003_check.npy',
    #             '../data/camera_calibration_frames_redo/to_copy_local_manual/1_1_2/1/000003_check.npy']
    # out_dir_calib = '../data/camera_calibration_frames_redo/to_copy_local_manual'

    cell_num = 2
    im_num = 567
    views = [0,3,1,2]
    file_dets = ['../data/camera_calibration_frames_redo/to_copy_local_manual/2_0_1_2_3/0/000000_check.npy',
                '../data/camera_calibration_frames_redo/to_copy_local_manual/2_0_1_2_3/1/000000_check.npy']
    out_dir_calib = '../data/camera_calibration_frames_redo/to_copy_local_manual'
    
    int_files = [os.path.join(out_dir_intrinsic,str(cell_num)+'_'+str(view)+'.npz') for view in views]
    

    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in range(63)]
    # 3]
    # ,1,2,3]
    # 0 1100,1400,1350,1800
    # 3 1100, 1400, 1600,2000
    out_dir = '../scratch/center_box'
    util.mkdir(out_dir)

    # check_rows = 7
    # check_cols = 9
    objp = np.zeros((check_cols*check_rows,3), np.float32)
    objp[:,:2] = np.mgrid[0:check_cols,0:check_rows].T.reshape(-1,2)
    print (objp[:20])

    # look at its detected checks
    for idx_view,view in enumerate(views):
        # print (im_num)
        im_path = get_im_path(meta_dir, interval_str, cell_num, view, im_num)
        intr = np.load(int_files[idx_view])
        # print (idx_view, view, int_files[idx_view])

        mtx = intr['mtx']
        dist = intr['dist']
        
        # print (idx_view, idx_view//2)
        corners = np.load(file_dets[idx_view//2])[idx_view%2,:,:]
        if cell_num==1:
            corners = np.reshape(corners,(check_rows,check_cols,2))
            corners = corners[::-1,:,:]
            corners = np.reshape(corners,(check_rows*check_cols,2))
            retval, rvec, tvec  =   cv2.solvePnP(objp[np.newaxis,:,:],  corners[np.newaxis,:,:], mtx, dist)
        elif view==0:
            load_dict = np.load(os.path.join(out_dir_calib,'1_'+str(view)+'.npz'))
            rvec_guess = load_dict['rvec']
            tvec_guess = load_dict['tvec']
            # print (rvec_guess,tvec_guess)
            retval, rvec, tvec  =   cv2.solvePnP(objp[np.newaxis,:,:],  corners[np.newaxis,:,:], mtx, dist, rvec_guess, tvec_guess, True)
        else:
            retval, rvec, tvec  =   cv2.solvePnP(objp[np.newaxis,:,:],  corners[np.newaxis,:,:], mtx, dist)

        
        # print (rvec)
        rot,jacob = cv2.Rodrigues(rvec)
        # print (rot)
        # print (euler_angles_from_rotation_matrix(rot))
        # print (tvec)
        
        camera_pos = np.matmul(-rot.T,tvec)
        print (view)
        print (camera_pos)
        print ('')

        out_file = os.path.join(out_dir_calib,'_'.join([str(val) for val in [cell_num,view]])+'.npz')
        print (out_file)
        np.savez(out_file,rvec = rvec, tvec= tvec)

        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp[np.newaxis,:,:], corners[np.newaxis,:,:], (w,h), mtx, None, flags = cv2.CALIB_USE_INTRINSIC_GUESS)
        # cv2.calibrateCamera(objpoints, imgpoints, (w,h), mtx, None, flags = cv2.CALIB_USE_INTRINSIC_GUESS)
        
        
        
        if viz:
            img = cv2.imread(im_path)
            # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # detect chessboard
            # ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows))
        
            out_file = os.path.join(out_dir,'_'.join([str(val) for val in [cell_num,view,check_rows,check_cols]])+'.jpg')
            # Draw and display the corners
            # dst = cv2.drawChessboardCorners(img, (check_cols,check_rows), corners,ret)
            
            for idx_pt, pt in enumerate(corners):
                dst = cv2.circle(img,tuple(np.int32(pt)),2,(255,0,int(idx_pt*255/63.)),-1)
            print('saving',out_file)
            cv2.imwrite(out_file,dst)


    # copy and block
    # detect
    # make sure first and last points are same across views
    # calibrate
    # cv2.calibrateCamera(objpoints, imgpoints, (w,h), mtx, None, flags = cv2.CALIB_USE_INTRINSIC_GUESS)

    # look at rotation matrices
    # share with sofia
    # ask johan to do what's needed with a bigger board

def viz_chess(im_path, check_path, just_two = False, out_file = None):

    img = cv2.imread(im_path)
    # out_file = os.path.join(out_dir,'_'.join([str(val) for val in [cell_num,view,check_rows,check_cols]])+'.jpg')
    # Draw and display the corners
    # dst = cv2.drawChessboardCorners(img, (check_cols,check_rows), corners,ret)
    if type(check_path)==str:
        c = np.load(check_path)
        print (c.shape)
        s = input()
        corners = np.load(check_path).squeeze()
    else:
        corners = check_path
    
    if just_two:
        corners = corners[[0,-1]]

    for idx_pt, pt in enumerate(corners):
        dst = cv2.circle(img,tuple(np.int32(pt)),10,(0,int(idx_pt*255/float(corners.shape[0])),0),-1)
    # print('saving',out_file)
    if out_file is not None:
        cv2.imwrite(out_file,dst)
    else:
        return img[:,:,::-1]


def script_check_chess_order():
    
    meta_dir = '../data/camera_calibration_frames_try2'
    out_dir_chess_det = os.path.join(meta_dir, 'common_im_withChessboardDet')
    interval_str = '20200428104445_113700'
    str_replace = [meta_dir,out_dir_chess_det.replace('..','/gross_pain')]

    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_html = os.path.join(meta_dir,'common_im_withChessboardDet_html')
    util.mkdir(out_dir_chess_det)
    util.mkdir(out_dir_html)

    out_dir_viz = os.path.join('../scratch','check_ord')
    util.mkdir(out_dir_viz)

    global check_cols, check_rows
    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict.keys():
        # print (key_curr, len(key_curr))
        if len(key_curr)<4:
            continue
        
        im_cols = common_im_dict[key_curr]
        if len(im_cols[0])==0:
            continue
        print (im_cols)
        s = input()
        for idx_im in range(len(im_cols[0])):
            for idx_view in range(len(im_cols)):
                im_file = im_cols[idx_view][idx_im]
                chess_file = im_file.replace(meta_dir, out_dir_chess_det).replace('.jpg','.npy')

                out_file = os.path.join(out_dir_viz,'_'.join(['%02d'%val for val in [idx_view,idx_im]])+'.jpg')
                viz_chess(im_file, chess_file, out_file = out_file)
                chess = np.load(chess_file).squeeze()
                chess = chess[::-1,:]
                out_file = os.path.join(out_dir_viz,'_'.join(['%02d'%val for val in [idx_view,idx_im]])+'_flip.jpg')
                viz_chess(im_file, chess, out_file = out_file)
            visualize.writeHTMLForFolder(out_dir_viz,height = 380, width = 672)
            break
        break
        # print (len(im_cols),len(im_cols[0]))
        # for idx_im_col, im_col in enumerate(im_cols):
        #     # print (len(im_col))
        #     for idx_im_file, im_file in enumerate(im_col):
        #         # print (key_curr, idx_im_col, idx_im_file, len(im_col))
        #         chess_file = im_file.replace(meta_dir, out_dir_chess_det).replace('.jpg','.npy')
        #         out_file = '../scratch/check_ord.jpg'
        #         viz_chess(im_file, chess_file, out_file = out_file)
        #         print (out_file)
        #         s = input()


def get_calib_im_lists(calib_im_file, meta_dir, interval_str):
    # meta_dir = '../data/camera_calibration_frames_try2'
    # interval_str = '20200428104445_113700'

    # calib_im_file = '../metadata/select_ims'
    lines = util.readLinesFromFile(calib_im_file)
    key_curr = None

    calib_im = {}
    for line in lines:
        if '.html' in line:
            # if key_curr is not None:
            #     print (key_curr)
            #     print (calib_im[key_curr])

            key_curr = tuple([int(val) for val in line[:-5].split('_')])
            calib_im[key_curr] = []
        else:
            # print (line)
            assert line.endswith('.jpg')
            im = line[:-4].split('_')
            im_num = int(im[-1])
            list_im = []
            for idx in range(len(key_curr)-1):
                im_path = get_im_path(meta_dir, interval_str, key_curr[0], key_curr[idx+1], im_num)
                # print (im_path)
                assert os.path.exists(im_path)
                list_im.append(im_path)
            calib_im[key_curr].append(list_im)

    return calib_im


def script_calibrate_big_board():
    # from checkerboard import detect_checkerboard
    # select an image

    meta_dir = '../data/camera_calibration_frames_try2'
    interval_str = '20200428104445_113700'
    out_dir_dets = os.path.join(meta_dir, 'calib_im_fix_dets')
    out_dir_intrinsic = os.path.join(meta_dir, 'intrinsics')
    out_dir_calib =  os.path.join(meta_dir, 'extrinsics')
    util.mkdir(out_dir_calib)
    
    cell_num = 2
    views = [0,1,2,3]

    dets = glob.glob(os.path.join(out_dir_dets, 'cell'+str(cell_num), interval_str, '*','*.npy'))
    im_nums = [int(os.path.split(det)[1][:-4].split('_')[-1]) for det in dets]
    im_nums = list(set(im_nums))
    print (im_nums)
    return

    # print (len(dets))
    # for det in dets:
    #     print (det)


    # file_dets = ['../data/camera_calibration_frames_redo/to_copy_local_manual/2_0_1_2_3/0/000000_check.npy',
    #             '../data/camera_calibration_frames_redo/to_copy_local_manual/2_0_1_2_3/1/000000_check.npy']
    # out_dir_calib = '../data/camera_calibration_frames_redo/to_copy_local_manual'
    
    # int_files = [os.path.join(out_dir_intrinsic,str(cell_num)+'_'+str(view)+'.npz') for view in views]
    
    print (check_rows, check_cols)
    objp = []
    for row in range(check_rows):
        for col in range(check_cols):
            objp.append([row,col, 0])
    objp = np.array(objp,dtype= np.float)
    objp = objp[:,np.newaxis,:]
    print (objp.shape)

    for im_num in im_nums:
        for idx_view,view in enumerate(views):
            det_path = get_im_path(out_dir_dets, interval_str, cell_num, view, im_num).replace('.jpg','.npy')
            if not os.path.exists(det_path):
                continue
            corners = np.load(det_path)[:,np.newaxis,:]
            print (im_num, view, corners.shape)
    
            int_file = os.path.join(out_dir_intrinsic,str(cell_num)+'_'+str(view)+'.npz')
            intr = np.load(int_file)
        
            mtx = intr['mtx']
            dist = intr['dist']
            
            retval, rvec, tvec  =   cv2.solvePnP(objp,  corners, mtx, dist)

            rot,jacob = cv2.Rodrigues(rvec)

            camera_pos = np.matmul(-rot.T,tvec)
            print (view)
            print (camera_pos)
            print ('')

            out_file = os.path.join(out_dir_calib,'_'.join([str(val) for val in [cell_num, view, im_num]])+'.npz')
            print (out_file)
            np.savez(out_file,rvec = rvec, tvec= tvec)



def script_format_for_bundler():
    meta_dir = '../data/camera_calibration_frames_try2'
    interval_str = '20200428104445_113700'
    out_dir_dets = os.path.join(meta_dir, 'calib_im_fix_dets')
    out_dir_intrinsic = os.path.join(meta_dir, 'intrinsics')
    out_dir_calib =  os.path.join(meta_dir, 'extrinsics')

    cell_num = 2
    im_num_select = 2976
    numtopicks = [0,1]

    cell_num = 1
    numtopicks = [2,0]

    views = [0,1,2,3]
    common_view = 3
    
    # get all image sets
    dets = glob.glob(os.path.join(out_dir_dets, 'cell'+str(cell_num), interval_str, '*','*.npy'))
    views_all = [int(os.path.split(det)[1][:-4].split('_')[-2]) for det in dets]
    im_nums_all = [int(os.path.split(det)[1][:-4].split('_')[-1]) for det in dets]
    im_nums = list(set(im_nums_all))
    # print (im_nums)

    # get their view sets
    view_sets = [[] for r in range(len(im_nums))]
    for idx_im_num, im_num in enumerate(im_nums_all):
        view_curr = views_all[idx_im_num]
        idx_im_nums = im_nums.index(im_num)
        view_sets[idx_im_nums].append(view_curr)
        view_sets[idx_im_nums].sort()
    
    # print (view_sets)
    # return

    # get board points 
    objp = []
    for row in range(check_rows):
        for col in range(check_cols):
            objp.append([row,col, 0])
    objp = np.array(objp,dtype= np.float)
    # print ('objp.shape',objp.shape)
    num_pts = objp.shape[0]

    # set up camera camera_ind, point_ind, and points_2d
    camera_ind = []
    point_ind = []
    points_2d = []
    points_3d = []

    for im_num, view_set in zip(im_nums, view_sets):
        # print (point_ind)
        # s = input()
        points_3d.append(np.array(objp))
        for view in view_set:
            point_ind += [(len(points_3d)-1)*num_pts+val for val in range(num_pts)]
        
            det_file = get_im_path(out_dir_dets, interval_str, cell_num, view, im_num).replace('.jpg','.npy')
            # print (det_file)
            dets = np.load(det_file)
            # print (dets.shape)
            assert dets.shape[0]==num_pts
            points_2d.append(dets)
            camera_ind += [view for val in range(num_pts)]
    points_2d = np.concatenate(points_2d, axis = 0)
    [point_ind, camera_ind ] = [np.array(val) for val in [point_ind, camera_ind]]
    # points_3d = np.array(points_3d)
    # print (points_3d.shape, points_2d.shape, point_ind.shape, camera_ind.shape)
    # s = input()

    # print (points_3d[0][:10])
    # transform obj pt to common view for each image set
    # transform obj pt sets to selected image set world coordinate frame (cam 3 is the origin)
    for idx_im_num, im_num in enumerate(im_nums):
        vals_file = [cell_num,common_view, im_num]        
        ext_file = os.path.join(out_dir_calib,'_'.join([str(val) for val in vals_file])+'.npz')
        loaded = np.load(ext_file)
        [rvec, tvec] = [loaded[key] for key in ['rvec','tvec']]
        rot, jacob = cv2.Rodrigues(rvec)
        pts_curr = points_3d[idx_im_num].T
        pts_curr = np.matmul(rot, pts_curr)+tvec
        pts_curr = pts_curr.T
        # print (pts_curr.shape)
        points_3d[idx_im_num] = pts_curr


    # print (points_3d[0][:10])

    # # transform obj pt sets to selected image set world coordinate frame
    tforms = [[] for r in views]

    for numtopick in numtopicks:
        im_num_curr = im_nums[numtopick]
        view_set_curr = [val for val in view_sets[numtopick] if val is not common_view]
        if numtopick==1:
            view_set_curr = [0]

        vals_file = [cell_num,common_view, im_num_curr]        
        ext_file = os.path.join(out_dir_calib,'_'.join([str(val) for val in vals_file])+'.npz')
        loaded = np.load(ext_file)
        [rvec0, tvec0] = [loaded[key] for key in ['rvec','tvec']]
        rot_inv = cv2.Rodrigues(cv2.Rodrigues(rvec0)[0].T)[0]
        print (view_set_curr)
        for view in view_set_curr:
            vals_file = [cell_num,view, im_num_curr]        
            ext_file = os.path.join(out_dir_calib,'_'.join([str(val) for val in vals_file])+'.npz')
            loaded = np.load(ext_file)
            [rvec, tvec] = [loaded[key] for key in ['rvec','tvec']]
            rot_inv_curr = cv2.Rodrigues(cv2.Rodrigues(rvec)[0].T)[0]
            rvecf,tvecf, _, _, _, _, _, _, _, _ =  cv2.composeRT(rvec*0, -tvec0,rot_inv, tvec0*0) 
            rvect, tvect, _, _, _, _, _, _, _, _ = cv2.composeRT(rvecf, tvecf,rvec, tvec) 

            tforms[view]=[rvect,tvect]
    tforms[common_view] = [tforms[common_view-1][0]*0,tforms[common_view-1][1]*0]
    tforms = np.array(tforms)
    tforms = np.reshape(tforms, (tforms.shape[0],-1))

    intrinsics = []
    # format intrinsics
    for view in views:
        int_file = os.path.join(out_dir_intrinsic,str(cell_num)+'_'+str(view)+'.npz')
        intr = np.load(int_file)
        mtx = intr['mtx']
        dist = intr['dist']
        mtx = mtx.ravel()[[0,2,4,5]]
        ints = np.concatenate([mtx,dist.ravel()])
        intrinsics.append(ints)
        # intrinsics.append([mtx,dist])

    intrinsics = np.array(intrinsics)
    camera_params = np.concatenate([tforms,intrinsics], axis = 1)
    
    for idx_im_num, im_num in enumerate(im_nums):
        for view in view_sets[idx_im_num]:
            im_pts_ac = np.load(get_im_path(out_dir_dets, interval_str, cell_num, view, im_num).replace('.jpg','.npy'))

            cam = camera_params[view]
            [rvec,tvec] = [vec[:,np.newaxis] for vec in [cam[:3],cam[3:6]]]
            mtx = np.eye(3).ravel()
            mtx[[0,2,4,5]] = cam[6:10]
            mtx = np.reshape(mtx,(3,3))
            dist = cam[10:15]
            dist = dist[np.newaxis,:]

            # (3, 1) (3, 1) (3, 3) (1, 5)
            im_pts, _ = cv2.projectPoints(points_3d[idx_im_num][:,np.newaxis,:], rvec, tvec, mtx, dist)
            # im_pts, _ = cv2.projectPoints(points_3d[idx_im_num][:,np.newaxis,:], tforms[view][0], tforms[view][1], intrinsics[view][0], intrinsics[view][1])

            im_pts = im_pts.squeeze()
            diffs = np.abs(im_pts_ac-im_pts)
            print (im_num, view, np.min(diffs), np.max(diffs))
    
    points_3d = np.concatenate(points_3d, axis = 0)
    # print (camera_ind[:100])
    # print (camera_ind[-100:])
    # print (camera_ind.shape)

    # print (points_3d.shape)
    # print (point_ind[-100:])
    # print (points_2d.shape)

    # save bundle
    out_file = os.path.join(out_dir_calib, str(cell_num)+'_bundle.npz')
    np.savez(out_file, points_3d = points_3d, points_2d = points_2d, point_ind = point_ind, camera_params = camera_params, camera_ind = camera_ind)





def main():
    script_format_for_bundler()
    # script_calibrate_big_board()
    # get_calib_gim_lists()
    # fix_filenames()
    # extract_calibration_vid_frames()
    # get_all_files_with_chessboard()
    # visualize_all_files_with_chessboard()

    # script_save_all_intrinsics()
    # save_common_im_with_chessboard_det()

    # script_stereo_calibrate()

    # script_calibrate_manual()

    # script_calibrate_center_board()

    # script_check_chess_order()

    
    # fname = '../data/camera_calibration_frames_try2/cell1/20200428104445_113700/0/ce_00_0_000998.jpg'
    # fname = '../data/camera_calibration_frames_try2/cell2/20200428104445_113700/0/ce_00_0_002897.jpg'
    # http://vision8.idav.ucdavis.edu:8000/gross_pain/data/camera_calibration_frames_try2/cell2/20200428104445_113700/0/ce_00_0_002897.jpg
    # print ()
    # get_img_chessboard(fname)
    return
    meta_dir = '../data/camera_calibration_frames_redo'
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_dets = os.path.join(meta_dir, 'chessboard_dets')
    out_dir_viz = os.path.join(meta_dir, 'chessboard_dets_viz')

    out_dir_copy = os.path.join(meta_dir,'to_copy_local_manual_2')
    util.mkdir(out_dir_copy)
    

    # common_im_dict = get_common_im(out_dir)
    # for key_curr in common_im_dict:
    #     if len(key_curr)<3:
    #         continue
    #     im_cols = common_im_dict[key_curr]
        
    #     if len(im_cols[0])==0:
    #         continue

    #     print (key_curr, len(im_cols), len(im_cols[0]))
    #     for idx_im_col, im_col in enumerate(im_cols):
    #         out_dir = os.path.join(out_dir_copy,'_'.join([str(val) for val in key_curr]), str(key_curr[idx_im_col+1]))
    #         util.makedirs(out_dir)
    #         for idx_im_path, im_path in enumerate(im_col):
    #             out_file = os.path.join(out_dir,'%06d'%idx_im_path+'.jpg')
    #             shutil.copyfile(im_path, out_file)

    # cell_num = str(1)
    # views = [str(val) for val in [1,2]]
    # file_nums = [166,283,656,271,293]
    cell_num = str(2)
    views = [str(val) for val in [0]]
    file_nums = [561]
    # out_dir = '1_0_3'

    for idx_file_num, file_num in enumerate(file_nums):
        out_dir = os.path.join(out_dir_copy,'_'.join([cell_num]+views))
        util.mkdir(out_dir)
        for view in views:
            out_dir_curr = os.path.join(out_dir,view)
            util.mkdir(out_dir_curr)
            in_dir = os.path.join(meta_dir,'cell'+cell_num,interval_str,view)
            # for idx_file_num,file_num in enumerate(file_nums):
            in_file = os.path.join(in_dir,'ce_00_'+view+'_%06d.jpg'%file_num)
            out_file = os.path.join(out_dir_curr,'%06d.jpg'%idx_file_num+1)
            print (in_file,out_file,os.path.exists(in_file))
            shutil.copyfile(in_file, out_file)


    print (out_dir_copy)







    
    
            
    


if __name__=='__main__':
    main()
