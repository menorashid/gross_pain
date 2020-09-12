import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
import pandas as pd
from helpers import util, visualize
from tqdm import tqdm

import cv2
import random

import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

import PIL
from PIL import Image

import multiprocessing

class TimeSplitter:
    def __init__(self, data_path, 
                str_aft ,
                str_aft_out , 
                horse_names = None, 
                frame_skip = 1):
        
        self.data_path = data_path
        # '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps'
        self.all_horse_names = ['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']

        if horse_names is None:
            self.horse_names = self.all_horse_names[:]    
        else:
            self.horse_names = horse_names
        
        self.str_aft = str_aft
        self.str_aft_out = str_aft_out

        self.frame_skip = frame_skip

    def add_time_index(self):
        data_path = self.data_path
        str_aft = self.str_aft
        horse_name = self.horse_names[0]
        for horse_name in self.horse_names:
            csv_path = os.path.join(data_path, horse_name+str_aft)
            df = pd.read_csv(csv_path)
            
            columns = [val for val in df.columns.values if 'Unnamed' not in val]
            df = df[columns]
            # columns.append('segment_ind')
            # print (df[:10])
            # print (columns)
            # s = input()
            # print (df[:10])
            new_df = []

            uni_intervals = df.interval.unique()
            for idx_interval, interval in enumerate(uni_intervals):
                for view_curr in df.view.unique():
                    rel_frames = df.loc[(df.interval==interval) & (df.view == view_curr)]
                    interval_ind = rel_frames.interval_ind.values[0]

                    rel_frames = rel_frames.sort_values('frame')
                    rel_frames = rel_frames.values
                    rel_frames = np.concatenate((rel_frames,-1*np.ones((rel_frames.shape[0],1))),axis = 1)
                    # str_arr = np.array(['']*len(rel_frames))[:,np.newaxis]
                    str_arr = np.array([-1]*len(rel_frames))[:,np.newaxis]
                    rel_frames = np.concatenate((rel_frames,str_arr),axis = 1)
                    
                    frame_inds = rel_frames[:,columns.index('frame')]
                    
                    diffs = frame_inds[1:] - frame_inds[:-1]
                    bin_breaks = np.where(diffs>self.frame_skip)[0]+1
                    bin_breaks = np.array([0]+list(bin_breaks)+[len(frame_inds)])
                    assert len(bin_breaks)<1000
                    for idx_seg, idx in enumerate(bin_breaks[:-1]):
                        start_idx = idx
                        end_idx =bin_breaks[idx_seg+1]
                        rel_frames[start_idx:end_idx,-2] = idx_seg
                        horse_idx = self.all_horse_names.index(horse_name)+1
                        seg_key = ''.join(['%1d'%horse_idx,'%01d'%view_curr,'%02d'%interval_ind,'%03d'%idx_seg])
                        rel_frames[start_idx:end_idx,-1] = int(seg_key)
                        # key_arr = [seg_key]*len(rel_frames)
                        # key_arr = np.array(key_arr)[:,np.newaxis]
                        # print (rel_frames.shape, key_arr.shape)
                        # rel_frames = np.concatenate((rel_frames,key_arr), axis = 1)
                        # print (rel_frames.shape)
                        # print (rel_frames[start_idx:start_idx+10])
                        # s = input()
                        # rel_frames[start_idx:end_idx,-1] = '_'.join([horse_name[:2],view_curr,interval_ind,idx_seg])
                    
                    
                    assert (np.sum(rel_frames[:,-2]<0)==0)
                    new_df.append(rel_frames)
                    # print (new_df[:10])
                    # s = input()

            columns.append('segment_ind')
            columns.append('segment_key')
            df = pd.DataFrame(new_df[0], columns=columns)
            for n_df in new_df[1:]:
                df = df.append(pd.DataFrame(n_df, columns = columns), ignore_index = True)
            out_file_csv = os.path.join(data_path, horse_name+self.str_aft_out)

            df.to_csv(path_or_buf= out_file_csv)
            print ('saved', out_file_csv, len(df))
            

                    


def main():
    print ('hello')
    data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    # str_aft = '_thresh_0.70_frame_index.csv'
    # str_aft = '_percent_0.01_frame_index.csv'
    str_aft = '_reduced_2fps_frame_index.csv'
    str_aft_out = '_reduced_2fps_frame_index_withSegIndexAndIntKey.csv'
    ts = TimeSplitter(data_path, str_aft, str_aft_out, frame_skip = 5) 
    ts.add_time_index()
    

if __name__=='__main__':
    main()