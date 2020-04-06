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

    dpi = 80
    margin = 0.001 # (5% of the width/height of the figure...)
    xpixels, ypixels = 2016, 1140

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = xpixels / dpi, ypixels / dpi

    ims = []
    figs = []
    axs = []
    pts = []
    for im_file in im_files:
        ims.append(Image.open(im_file))
        fig = plt.figure()
        # figsize=figsize, dpi=dpi)
        # ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
        figs.append(fig)
        # axs.append(ax)
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
    # dirs = ['_'.join([str(val) for val in [cell,view1,view2]]) for cell in range(1,3) for view1 in range(4) for view2 in range(view1+1,4)]
    meta_dir = '../data/to_copy_local_manual'
    # dirs = [dir_curr for dir_curr in glob.glob(os.path.join(meta_dir,'*')) if os.path.isdir(dir_curr)]
    
    # ax.imshow(np.random.random((xpixels, ypixels)), interpolation='none')
    # plt.show()
    dir_curr = os.path.join(meta_dir,'1_0_3')
    sessions = [1,1,3,1,1]
    
    # [3,1,1]

    # for dir_curr in dirs:
    views = os.path.split(dir_curr)[1].split('_')[-2:]
    im_files = [os.path.split(file_curr)[1] for file_curr in glob.glob(os.path.join(dir_curr,views[0],'*.jpg'))]
    im_files.sort()

    im_files = im_files[3:4]
    sessions = sessions[3:4]
    print im_files


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
