import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
from helpers import util, visualize
import multiprocessing
import subprocess
import cv2
import pandas as pd
import time

class OpticalFlowExtractor():

    def __init__(self,output_dir, output_flow, num_processes = 1, min_flow = -15, max_flow = 15):
        self.output_dir = output_dir
        self.min = min_flow
        self.max = max_flow
        self.mag_max = np.sqrt(np.square(self.min)+np.square(self.max))
        self.output_flow = output_flow
        self.num_processes = num_processes
        self.subjects = ['aslan' , 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']

    def get_im_path(self, row):
        im_path = util.get_image_name(row['subject'], row['interval_ind'], row['interval'], row['view'], row['frame'], self.output_dir)
        if not os.path.exists(im_path):
            im_path = None
        return im_path
    
    def get_opt_flow(self, args):
        (out_file, first, second) = args
        
        first = cv2.imread(first,cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(second,cv2.IMREAD_GRAYSCALE)
        
        flow = cv2.calcOpticalFlowFarneback(first,second, None, 0.5, 1, 15, 3, 5, 1.2, 0)
        
        flow = np.clip(flow, self.min, self.max)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        flow = ((flow - self.min)/(self.max - self.min))
        mag = mag/self.mag_max
        
        str_flow = '%.5f'%(np.max(mag))
        out_file_text = out_file[:-4]+'.txt'
        util.writeFile(out_file_text,[str_flow])

        flow = np.concatenate((flow, mag[:,:,np.newaxis]), axis=2)*255
        flow = np.clip(flow, 0, 255)
        flow = flow.astype(np.uint8)
        cv2.imwrite(out_file, flow)


    def extract_frames(self, replace = True, subjects_to_extract = None):
        
        if subjects_to_extract is None:
            subjects_to_extract = self.subjects
        
        # To help us read the data, we save a long .csv-file
        # with labels for each frame(a frame index).
        column_headers = ['interval', 'interval_ind', 'view', 'subject', 'pain', 'frame']
        
        for i, subject in enumerate(subjects_to_extract):

            out_file_index = os.path.join(self.output_dir,subject+'_'+'frame_index.csv')
            frames = pd.read_csv(out_file_index)
            rel_intervals = frames.interval.unique()
            rel_views = frames.view.unique()
            for idx_interval,interval in enumerate(rel_intervals):
                    
                for view in rel_views:
                    rel_frames = frames.loc[(frames['interval'] == interval) & (frames['view'] == view)]
                    rel_frames = rel_frames.sort_values('frame')
                    print (len(rel_frames))
                    # print (rel_frames)
                    # rel_frames = rel_frames.reindex()
                    # print (rel_frames)
                    args = []
                    for idx in range(len(rel_frames)-1):
                        first = self.get_im_path(rel_frames.iloc[idx])
                        second = self.get_im_path(rel_frames.iloc[idx+1])
                        if (first is None) or (second is None):
                            continue

                        out_file = os.path.split(second)
                        out_dir = out_file[0]+'_opt'
                        out_dir = out_dir.replace(self.output_dir, self.output_flow)
                        out_file = os.path.join(out_dir,out_file[1].replace('.jpg','.png'))
                        if not replace and os.path.exists(out_file):
                            continue
                        util.makedirs(os.path.split(out_file)[0])
                        args.append((out_file, first, second))
                            # , self.mag_max, self.min, self.max))

                    t = time.time()
                    print ('doing interval number {} out of {}, view {}, num frames {}'.format(idx_interval, len(rel_intervals), view, len(args)))
                    # args = args[:10]
                    # for arg in args:
                    #     self.get_opt_flow(arg)
                    pool = multiprocessing.Pool(self.num_processes)
                    pool.map(self.get_opt_flow,args)
                    pool.close()
                    pool.join()

                    print ('done with interval number {} out of {}, view {}, num frames {}, time taken {}'.format(idx_interval, len(rel_intervals), view, len(args),time.time()-t ))
                    visualize.writeHTMLForFolder(out_dir, ext = '.png')
                    # break
                # break
            # break
        

def main():
    fps = 10
    width = 672
    height = 380
    str_dir = '_'.join([str(val) for val in [width,height,fps]])
    out_dir_testing = '../data/pain_no_pain_x2h_intervals_for_extraction_'+str_dir+'fps'
    out_dir_flow = '../data_other/pain_no_pain_x2h_intervals_for_extraction_'+str_dir+'fps'
    util.mkdir(out_dir_testing)

    ofe = OpticalFlowExtractor(output_dir = out_dir_testing, num_processes = multiprocessing.cpu_count(), output_flow = out_dir_flow)    
    ofe.extract_frames(replace = False, subjects_to_extract = None)


if __name__=='__main__':
    main()



