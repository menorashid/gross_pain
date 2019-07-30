import os
from helpers import util, visualize
import cv2
import numpy as np
import glob

def main():

    dir_data = '../data/tester_kit/naughty_but_nice'
    out_dir_meta = '../scratch/polyline_test'
    util.mkdir(out_dir_meta)

    vid_names = ['ch01_20181209092600'  ,'ch03_20181209092844'  ,'ch05_20181208230458'  ,'ch06_20181209093714']
    
    vid_name = vid_names[0]
    contour_len = []
    for vid_name in vid_names:
        out_dir_curr = os.path.join(out_dir_meta, vid_name)
        util.mkdir(out_dir_curr)

        dir_anno = os.path.join(dir_data, vid_name+'_anno')
        mask_files = glob.glob(os.path.join(dir_anno, '*.png'))
        mask_files.sort()

        # mask_file = mask_files[0]
        for mask_file in mask_files:
            img = cv2.imread(mask_file,cv2.IMREAD_COLOR)
            ret,thresh = cv2.threshold(img,127,255,0)
            contours, hierarchy = cv2.findContours(thresh[:,:,0], cv2.RETR_EXTERNAL, 2)
            img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
            out_file_contours = os.path.join(out_dir_curr, os.path.split(mask_file)[1])
            cv2.imwrite(out_file_contours, img)
            if len(contours)>0:
                contour_len.append(len(contours[0]))
            # cnt = contours[0]
            # print len
            # print out_file_contours
            # , len(cnt)
        
        visualize.writeHTMLForFolder(out_dir_curr, height = 256, width = 448, ext = '.png')
        print np.min(contour_len), np.max(contour_len), np.mean(contour_len)
# perimeter = cv2.arcLength(cnt,True)
# epsilon = 0.1*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)



    print ('hello')


if __name__=='__main__':
    main()
