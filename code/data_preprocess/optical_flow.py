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
from tqdm import tqdm
import glob

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

    def get_im_dir(self, row, meta_dir):
        frame_dir = os.path.split(self.get_im_path(row))[0]
        frame_dir = frame_dir.replace(self.output_dir, meta_dir)
        return frame_dir
    
    def get_flow_path_from_rgb(self,rgb_path):
        out_file = os.path.split(rgb_path)
        out_dir = out_file[0]+'_opt'
        out_dir = out_dir.replace(self.output_dir, self.output_flow)
        out_file = os.path.join(out_dir,out_file[1].replace('.jpg','.png'))
        return out_file

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

                        out_file = self.get_flow_path_from_rgb(second)
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
    
    def add_symlinks(self, subjects_to_extract = None):
        assert self.output_dir is not self.output_flow
        real_out = os.path.realpath(self.output_dir)
        real_flow = os.path.realpath(self.output_flow)

        if subjects_to_extract is None:
            subjects_to_extract = self.subjects
        
        for i, subject in enumerate(subjects_to_extract):

            out_file_index = os.path.join(self.output_dir,subject+'_'+'frame_index.csv')
            frames = pd.read_csv(out_file_index)
            rel_intervals = frames.interval.unique()
            rel_views = frames.view.unique()
            for idx_interval,interval in enumerate(rel_intervals):
                for view in rel_views:
                    rel_frames = frames.loc[(frames['interval'] == interval) & (frames['view'] == view)]
                    
                    flow_dir = self.get_im_dir(rel_frames.iloc[0], real_flow)
                    flow_dir = flow_dir+'_opt'

                    target_dir = flow_dir.replace(real_flow,real_out)
                    meta_target = os.path.split(target_dir)[0]

                    if not os.path.exists(flow_dir):
                        print ('flow dir does not exist:',flow_dir)
                        continue

                    if not os.path.exists(meta_target):
                        print ('meta_target dir does not exist:',meta_target)
                        continue

                    if os.path.exists(target_dir):
                        print ('target_dir alread exists:', target_dir)
                        continue

                    command = ['ln','-s', flow_dir, target_dir]
                    print (' '.join(command))

                    subprocess.call(command)
            #         break
            #     break
            # break

    def collate_magnitude(self, subjects_to_extract = None):
        if subjects_to_extract is None:
            subjects_to_extract = self.subjects
        
        # To help us read the data, we save a long .csv-file
        # with labels for each frame(a frame index).
        # column_headers = ['interval', 'interval_ind', 'view', 'subject', 'pain', 'frame','max_flow_mag']
        for i, subject in enumerate(subjects_to_extract):
            print ('Subject: {}'.format(subject))
            out_file_index = os.path.join(self.output_dir,subject+'_'+'frame_index.csv')
            frames = pd.read_csv(out_file_index)
            rel_intervals = frames.interval.unique()
            rel_views = frames.view.unique()
            for idx_interval,interval in enumerate(rel_intervals):
                commands = []
                for view in rel_views:
                    rel_frames = frames.loc[(frames['interval'] == interval) & (frames['view'] == view)]
                    flow_dir = self.get_im_dir(rel_frames.iloc[0], self.output_dir)
                    flow_dir = flow_dir+'_opt'
                    assert os.path.exists(flow_dir)

                    out_file_files = flow_dir+'_files.txt'
                    out_file_mags = flow_dir+'_mags.txt'

                    if os.path.exists(out_file_mags):
                        print('continuing',out_file_mags)
                        continue

                    t = time.time()
                    txt_files = os.listdir(flow_dir)
                    txt_files = [os.path.join(flow_dir,file) for file in txt_files if file.endswith('.txt')]

                    util.writeFile(out_file_files,txt_files)
                    
                    command = ['grep','-v',"'^#'",out_file_files,'|','xargs','cat','>>',out_file_mags]
                    
                    # this is the manual part. instead of the subprocess.
                    command = ' '.join(command)
                    # commands.append(command)
                    print (command)
                    subprocess.call(command, shell = True)
                    
                    print ('done with interval number {} out of {}, view {},  time taken {}'.format(idx_interval, len(rel_intervals), view, time.time()-t ))

                # util.writeFile(os.path.join(os.path.split(flow_dir)[0],'commands.txt'),commands)

    def create_thresholded_csv(self, thresh, subjects_to_extract = None, percent = False):
        if subjects_to_extract is None:
            subjects_to_extract = self.subjects
        
        # To help us read the data, we save a long .csv-file
        # with labels for each frame(a frame index).
        column_headers = ['interval', 'interval_ind', 'view', 'subject', 'pain', 'frame']

        for i, subject in enumerate(subjects_to_extract):
            
            print ('Subject: {}'.format(subject))
            in_file_index = os.path.join(self.output_dir,subject+'_'+'frame_index.csv')
            str_thresh = '%.2f'%thresh
            if percent:
                out_file_index = os.path.join(self.output_dir,subject+'_percent_'+str_thresh+'_'+'frame_index.csv')
            else:
                out_file_index = os.path.join(self.output_dir,subject+'_thresh_'+str_thresh+'_'+'frame_index.csv')
            print (out_file_index)
            frames = pd.read_csv(in_file_index)
            rel_intervals = frames.interval.unique()
            rel_views = frames.view.unique()
            
            big_list = []

            for idx_interval,interval in enumerate(rel_intervals):
                
                fids = []
                for view in rel_views:
                    rel_frames = frames.loc[(frames['interval'] == interval) & (frames['view'] == view)]
                    
                    flow_dir = self.get_im_dir(rel_frames.iloc[0], self.output_dir)
                    flow_dir = flow_dir+'_opt'
                    out_file_files = flow_dir+'_files.txt'
                    out_file_mags = flow_dir+'_mags.txt'

                    if not os.path.exists(out_file_files):
                        print('continuing',out_file_files)
                        continue

                    files = np.array(util.readLinesFromFile(out_file_files))
                    mags = [float(val) for val in util.readLinesFromFile(out_file_mags)]
                    mags = np.array(mags)
                    assert len(files)==len(mags)

                    if percent:
                        idx_sort = np.argsort(mags)
                        num_keep = int(len(mags)*thresh)
                        bin_keep = idx_sort[-num_keep:]
                        # print (len(mags),num_keep)
                        # print (mags[bin_keep[:10]])
                        # bin_keep = idx_sort[:num_keep]
                        # print (mags[bin_keep[:10]])
                        # print (out_file_index)
                        # s = input()
                    else:
                        bin_keep = mags>=thresh
                    files_keep = files[bin_keep]
                    print ('view,len(files), len(files_keep)',view,len(files), len(files_keep))
                    fids_curr = [int(os.path.split(file)[1][:-4].split('_')[-1]) for file in files_keep]
                    fids +=fids_curr

                # unique_fids = list(set(fids))
                # fids_keep = []
                # for fid in unique_fids:
                #     if fids.count(fid)==4:
                #         fids_keep.append(fid)
                # print (len(fids_keep))

                print (len(fids))
                fids = list(set(fids))
                print (len(fids))
                # s = input()
                row_in = rel_frames.iloc[0]
                for fid in fids:
                    for view in rel_views:
                        row_out = [row_in['interval'], row_in['interval_ind'], view, row_in['subject'], row_in['pain'], fid]
                        big_list.append(row_out)

            
            frame_index_df = pd.DataFrame(big_list, columns=column_headers)
            print(out_file_index)
            print (len(big_list))
            frame_index_df.to_csv(path_or_buf= out_file_index)



def main():
    fps = 10
    width = 672
    height = 380
    str_dir = '_'.join([str(val) for val in [width,height,fps]])
    out_dir_testing = '../data/pain_no_pain_x2h_intervals_for_extraction_'+str_dir+'fps'
    out_dir_flow = '../data_other/pain_no_pain_x2h_intervals_for_extraction_'+str_dir+'fps'
    util.mkdir(out_dir_testing)

    ofe = OpticalFlowExtractor(output_dir = out_dir_testing, num_processes = multiprocessing.cpu_count(), output_flow = out_dir_flow)    

    # # extract optical flow
    # # ofe.extract_frames(replace = False, subjects_to_extract = None)

    # # add symlinks if storing optical flow in a different drive
    # ofe.add_symlinks(subjects_to_extract = None)

    # # collate magnitudes and files names in text files. need to manually run commands files after this step.
    # ofe.collate_magnitude(subjects_to_extract = ['inkasso'])
    # ['julia', 'kastanjett', 'naughty_but_nice', 'sir_holger'])

    # create csv with thresholded images only
    # ofe.create_thresholded_csv(thresh = 0.7,subjects_to_extract = ['inkasso'])
    ofe.create_thresholded_csv(thresh = 0.01,subjects_to_extract = None, percent = True)
    # , 'kastanjett', 'naughty_but_nice', 'sir_holger'] )



if __name__=='__main__':
    main()



