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
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    

    out_dir_html = os.path.join(meta_dir,'common_im_html')
    str_replace = ['..','/gross_pain']
    util.mkdir(out_dir_html)

    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict.keys():

        im_cols = common_im_dict[key_curr]
        print (key_curr, len(im_cols))
        # im_cols = np.array(im_cols)
        if len(im_cols[0])>0:
            out_file_html = os.path.join(out_dir_html,'_'.join([str(val) for val in key_curr])+'.html')

            for idx_im_col in range(len(im_cols)):
                for idx_im in range(len(im_cols[idx_im_col])):
                    im_cols[idx_im_col][idx_im] = im_cols[idx_im_col][idx_im].replace(str_replace[0],str_replace[1])
            views = list(key_curr)[1:]
            captions = [[str(view)]*len(im_cols[idx_view]) for idx_view, view in enumerate(views)]
            visualize.writeHTML(out_file_html, im_cols, captions)

def save_common_im_with_chessboard_det():
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_chess_det = '../data/camera_calibration_frames_withChessboardDet'
    util.mkdir(out_dir_chess_det)
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    

    out_dir_html = os.path.join(meta_dir,'common_im_withChessboardDet_html')
    str_replace = [meta_dir,'/gross_pain/data/camera_calibration_frames_withChessboardDet']
    util.mkdir(out_dir_html)

    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict.keys():
        if len(key_curr)==2:
            continue
        
        im_cols = common_im_dict[key_curr]
        for im_col in im_cols:
            for im_file in im_col:
                out_file = im_file.replace(meta_dir, out_dir_chess_det)
                if os.path.exists(out_file):
                    continue
                util.makedirs(os.path.split(out_file)[0])
                

                check_cols = 7
                check_rows = 9
                img = cv2.imread(im_file)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # detect chessboard
                ret, corners = cv2.findChessboardCorners(gray, (check_cols,check_rows),cv2.CALIB_CB_FAST_CHECK)
                assert ret
                dst = cv2.drawChessboardCorners(img, (check_cols,check_rows), corners,ret)
                cv2.imwrite(out_file,dst)
        
        if len(im_cols[0])>0:
            out_file_html = os.path.join(out_dir_html,'_'.join([str(val) for val in key_curr])+'.html')
            for idx_im_col in range(len(im_cols)):
                for idx_im in range(len(im_cols[idx_im_col])):
                    im_cols[idx_im_col][idx_im] = im_cols[idx_im_col][idx_im].replace(str_replace[0],str_replace[1])
            views = list(key_curr)[1:]
            captions = [[' '.join([str(views[idx_im_col]),os.path.split(im_file)[1][-10:-4]]) for im_file in im_col] for idx_im_col,im_col in enumerate(im_cols)]
            # captions = [[str(view)]*len(im_cols[idx_view]) for idx_view, view in enumerate(views)]
            visualize.writeHTML(out_file_html, im_cols, captions)

def script_save_all_intrinsics():
    meta_dir = '../data/camera_calibration_frames_redo'
    out_dir_chess_det = '../data/camera_calibration_frames_withChessboardDet'
    util.mkdir(out_dir_chess_det)
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_intrinsic = os.path.join(meta_dir,'intrinsics')
    util.mkdir(out_dir_intrinsic)

    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict:
        if len(key_curr)>2:
            continue

        im_files = common_im_dict[key_curr]
        assert len(im_files)==1
        im_files = im_files[0]
        # [:10]
        print (key_curr, len(im_files))
        # util.mkdir('../scratch/check_row_col')
        mtx, dist = do_intrinsic(im_files)
        # ,out_dir = '../scratch/check_row_col')

        print (mtx, type(mtx))
        print (dist, type(dist))
        out_file = os.path.join(out_dir,'_'.join([str(val) for val in key_curr])+'.npz')
        print (out_file)
        np.savez(out_file, mtx = mtx, dist = dist)
        # s = input()

def save_im_chessboard_dets(arg):
    (im_file, out_file, out_file_viz) = arg
    num_dets = 4 
    check_cols = 7 
    check_rows = 9
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


def main():
    # fix_filenames()
    # extract_calibration_vid_frames()
    # get_all_files_with_chessboard()
    # visualize_all_files_with_chessboard()
    # save_common_im_with_chessboard_det()


    meta_dir = '../data/camera_calibration_frames_redo'
    interval_str = '20200317130703_131800'
    out_dir = os.path.join(meta_dir, 'ims_to_keep')
    out_dir_dets = os.path.join(meta_dir, 'chessboard_dets')
    out_dir_viz = os.path.join(meta_dir, 'chessboard_dets_viz')

    out_dir_copy = os.path.join(meta_dir,'to_copy_local')
    util.mkdir(out_dir_copy)
    

    common_im_dict = get_common_im(out_dir)
    for key_curr in common_im_dict:
        if len(key_curr)<3:
            continue
        im_cols = common_im_dict[key_curr]
        
        if len(im_cols[0])==0:
            continue

        print (key_curr, len(im_cols), len(im_cols[0]))
        for idx_im_col, im_col in enumerate(im_cols):
            out_dir = os.path.join(out_dir_copy,'_'.join([str(val) for val in key_curr]), str(key_curr[idx_im_col+1]))
            util.makedirs(out_dir)
            for idx_im_path, im_path in enumerate(im_col):
                out_file = os.path.join(out_dir,'%06d'%idx_im_path+'.jpg')
                shutil.copyfile(im_path, out_file)

    print (out_dir_copy)







    
    
            
    


if __name__=='__main__':
    main()
