import matplotlib
matplotlib.use('TkAgg')

import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

import numpy as np
from PIL import Image
import glob
from helpers import util


def plot_and_save_correspondences(im_files,out_file,num_sessions=1 ):
    plt.ion()


    ims = []
    figs = []
    axs = []
    pts = []
    for im_file in im_files:
        ims.append(Image.open(im_file))
        fig = plt.figure()
        figs.append(fig)
        pts.append([])

    for idx_im, im in enumerate(ims):
        plt.figure(figs[idx_im].number)
        plt.imshow(im, interpolation = None)

    for idx_im, im in enumerate(ims):

        plt.figure(figs[idx_im].number)
        cursor = Cursor(plt.gca(), useblit = True, color='red', linewidth=1)

        for i in range(num_sessions):
            raw_input()
            pts[idx_im] += plt.ginput(-1,timeout = 0, show_clicks=True)
            print pts[idx_im]
            for pt in pts[idx_im]:
                plt.plot(pt[0],pt[1],'*b')    

    assert len(pts[0])==len(pts[1])
    pts = np.array(pts)
    print pts.shape
    np.save(out_file, pts)
    print 'saved',out_file

    plt.close('all')
    # [0]
    # print pt
    # pts[idx_im].append(pt)
    # plt.plot(pt[0],pt[1],'*r')

    raw_input()




def main():
    print 'hello'
    meta_dir = '../data/to_copy_local_manual'
    dir_curr = os.path.join(meta_dir,'1_0_3')
    sessions = [1,1,3,1,1]

    meta_dir = '../data/to_copy_local_manual_2'
    dir_curr = os.path.join(meta_dir,'2_0_1_2_3')
    sessions = [1]


    views = ['1','2']
    # os.path.split(dir_curr)[1].split('_')[-2:]
    im_files = [os.path.split(file_curr)[1] for file_curr in glob.glob(os.path.join(dir_curr,views[0],'*.jpg'))]
    im_files.sort()

    # im_files = im_files[3:4]
    # sessions = sessions[3:4]
    # print im_files


    assert len(im_files)==len(sessions)
    for num_session,im_file in zip(sessions, im_files):
        # im_file = im_files[-1]
        im_pair = [os.path.join(dir_curr,view,im_file) for view in views]
        assert os.path.exists(im_pair[1])
        
        out_file = im_pair[0].replace('.jpg','_check.npy')
        print im_file, im_pair, out_file, num_session 
        
        plot_and_save_correspondences(im_pair, out_file, num_session)






                 


    
    
            
    


if __name__=='__main__':
    main()
