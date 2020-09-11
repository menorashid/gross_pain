import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torchvision
import imageio
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from random import shuffle
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import io as rhodin_utils_io
from tqdm import tqdm
import time
import random
from multiview_dataset import get_label_df_for_subjects

class SegBasedSampler(Sampler):
    """ This sampler decides how to iterate over the indices in the dataset.
        Prepares batches of sub-batches, where a sub-batch contains
        indices corresponding to frames from different views at t,
        and indices corresponding to frames from different
        views at t', from the same interval."""

    def __init__(self, data_folder, save_path, mode,
                 batch_size,
                 num_frames_per_seg,
                 subjects=None,
                 randomize=True,
                 every_nth_segment=1,
                 str_aft = None,
                 min_size = 10,
                 ):
        
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        # build view/subject datastructure
        self.label_dict = get_label_df_for_subjects(data_folder, subjects, str_aft = str_aft)
        self.columns = ['segment_key']
        # ['subject','view','interval_ind','segment_ind']
        columns = self.columns

        df_select = self.label_dict[columns[0]].drop_duplicates().values.squeeze()
        # print (df_select.shape)
        # print (len(df_select))
        all_keys = []
        frame_idx_dict = {}
        num_skipped = 0
        num_frames = 0

        seg_key_vals = self.label_dict[self.columns[0]].values
        frames_vals = np.array(self.label_dict.index.tolist())
        # print (frames_vals[:10])
        # s = input()
        with tqdm(total=len(df_select)) as pbar:
            for idx, row in enumerate(df_select):
            # df_select.iterrows():
                pbar.update(1)
                key_full = row
                # tuple(row.values)
                # bin_select = np.ones((len(self.label_dict),))
                key = columns[0]
                bin_select = seg_key_vals== key_full
                # row[key]
                # for key in columns[1:]:
                #     print ('here')
                #     bin_select = bin_select & (self.label_dict[key]==row[key])

                # frames = self.label_dict.loc[bin_select]
                frames = frames_vals[bin_select]
                
                if len(frames)<self.min_size:
                    num_skipped +=1
                    continue
                all_keys.append(key_full)
                # frame_idx = list(frames.index)
                frame_idx = list(frames)

                num_frames += len(frame_idx)
                frame_idx_dict[key_full] = frame_idx
                # print (row)
                # print (self.label_dict.loc[frame_idx])
                # s = input()
            
        self.all_keys = all_keys
        self.all_keys.sort()
        self.all_keys = self.all_keys[::self.every_nth_segment]
        self.frame_idx_dict = frame_idx_dict


    def __len__(self):
        return len(self.label_dict['frame'])

    def add_segments_to_batch(self, frame_idx_list, batches, curr_batch):
        for frame_idx_curr in frame_idx_list:
            num_frames = len(frame_idx_curr)
            num_batch = len(curr_batch)

            if (num_batch+num_frames)<=self.batch_size:
                curr_batch+=frame_idx_curr
                # batches.append(curr_batch)
            else:
                batches.append(curr_batch)
                curr_batch = frame_idx_curr[:]
        return batches, curr_batch

    def __iter__(self):
        

        index_list = []
        print("Randomizing dataset (SegBasedSampler.__iter__)")
        s_time = time.time()
       
        batches = []
        curr_batch = []

        
        random.shuffle(self.all_keys)
        for index in range(len(self.all_keys)):
            
            key = self.all_keys[index]
            frame_idx = self.frame_idx_dict[key]
            num_frames = len(frame_idx)

            if num_frames>self.num_frames_per_seg:
                new_batches = [frame_idx[x:x+self.num_frames_per_seg] for x in range(0,num_frames,self.num_frames_per_seg)]
                
                if len(new_batches[-1])<self.min_size:
                    new_batches = new_batches[:-1]
                
                batches, curr_batch = self.add_segments_to_batch(new_batches[::-1],batches, curr_batch)
            else:
                batches, curr_batch = self.add_segments_to_batch([frame_idx], batches, curr_batch)

        return iter(batches)


def main():
    print ('hello')
    data_folder = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    all_subjects = ['aslan','brava','herrera','julia','kastanjett','naughty_but_nice','sir_holger']
    sampler = SegBasedSampler(data_folder, None, None,
                 1024,
                 num_frames_per_seg = 1024,
                 subjects = all_subjects,
                 randomize=True,
                 every_nth_segment=1,
                 str_aft = '_reduced_2fps_frame_index_withSegIndexAndKey.csv')
    print (sampler.subjects)
    for batches in sampler:
        print (len(batches))
        print (batches[::10])
        s = input()

if __name__=='__main__':
    main()

