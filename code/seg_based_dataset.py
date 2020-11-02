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
from multiview_dataset import MultiViewDataset, get_label_df_for_subjects
import copy

class SegBasedSampler(Sampler):
    """ This sampler decides how to iterate over the indices in the dataset.
        Prepares batches of sub-batches, where a sub-batch contains
        indices corresponding to frames from different views at t,
        and indices corresponding to frames from different
        views at t', from the same interval."""

    def __init__(self, data_folder, 
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
        columns = self.columns

        df_select = self.label_dict[columns[0]].drop_duplicates().values.squeeze()
        print (df_select.shape)

        all_keys = []
        frame_idx_dict = {}
        num_skipped = 0
        num_frames = 0

        seg_key_vals = self.label_dict[self.columns[0]].values
        frames_vals = np.array(self.label_dict.index.tolist())
        with tqdm(total=len(df_select)) as pbar:
            for idx, row in enumerate(df_select):
                pbar.update(1)
                key_full = row
                key = columns[0]
                bin_select = seg_key_vals== key_full
                frames = frames_vals[bin_select]
                
                if len(frames)<self.min_size:
                    num_skipped +=1
                    continue
                all_keys.append(key_full)
                frame_idx = list(frames)

                num_frames += len(frame_idx)
                frame_idx_dict[key_full] = frame_idx
            
        self.all_keys = all_keys
        self.all_keys.sort()
        self.all_keys = self.all_keys[::self.every_nth_segment]
        self.frame_idx_dict = frame_idx_dict
        self.batches = None

    def __len__(self):
        return len(self.label_dict['frame'])


    def get_pain_counts(self):
        counts = [0,0]
        for key in self.all_keys:
            num_frames = len(self.frame_idx_dict[key])
            # num_frames = 1210
            num_segs = num_frames//self.num_frames_per_seg
            rem = num_frames%self.num_frames_per_seg
            if rem>=self.min_size:
                num_segs += 1
            pain_label = self.label_dict.loc[self.label_dict['segment_key']==key,'pain'].iloc[0]
            counts[pain_label]+=num_segs

        return counts

    def get_pain_weight(self):
        train_counts = np.array(self.get_pain_counts())
        weights = 1/train_counts
        weights = weights/np.sum(weights) * 2
        return weights



    def add_segments_to_batch(self, frame_idx_list, batches, curr_batch, batch_keys, key):
        for frame_idx_curr in frame_idx_list:
            num_frames = len(frame_idx_curr)
            num_batch = len(curr_batch)

            if (num_batch+num_frames)<=self.batch_size:
                curr_batch+=frame_idx_curr
                batch_keys[-1].append(key)
                # batches.append(curr_batch)
            else:
                batches.append(curr_batch)
                curr_batch = frame_idx_curr[:]
                batch_keys.append([key])
        return batches, curr_batch, batch_keys

    def __iter__(self):
        
        if (self.batches is None) or (self.randomize):
            index_list = []
            # s_time = time.time()
           
            batches = []
            curr_batch = []
            batch_keys = [[]]
            print("Making dataset SegBasedSampler")
            if self.randomize:
                print("Randomizing dataset (SegBasedSampler.__iter__)")
                random.shuffle(self.all_keys)

            frame_idx_dict = copy.deepcopy(self.frame_idx_dict)
            all_keys = copy.deepcopy(self.all_keys)
            while len(all_keys)>0:

            # for index in range(len(self.all_keys)):
                
                # key = self.all_keys[index]
                key = all_keys.pop(0)
                
                # if key==1105001:
                #     print (len(frame_idx_dict[key]),frame_idx_dict[key][0])

                if key in batch_keys[-1]:
                    batches.append(curr_batch)
                    batch_keys.append([])
                    curr_batch = []

                frame_idx = frame_idx_dict[key]
                frame_idx.sort()


                
                num_frames = len(frame_idx)

                if num_frames>self.num_frames_per_seg:
                    frame_idx_keep = frame_idx[:self.num_frames_per_seg]
                    frame_idx_rest = frame_idx[self.num_frames_per_seg:]
                    if len(frame_idx_rest)>=self.min_size:
                        frame_idx_dict[key] = frame_idx_rest
                        all_keys.append(key)

                    batches, curr_batch, batch_keys = self.add_segments_to_batch([frame_idx_keep],batches, curr_batch, batch_keys, key)

                else:
                    batches, curr_batch, batch_keys = self.add_segments_to_batch([frame_idx], batches, curr_batch, batch_keys, key)


            batches.append(curr_batch)
            self.batch_keys = batch_keys
            self.batches = batches
            # print (self.batch_keys)
            if self.randomize:
                random.shuffle(self.batches)

        print ('HOLA', self.num_frames_per_seg, len(self.batch_keys))
        return iter(self.batches)


def main():
    print ('hello')
    import random
    # torch.random.seed(2)
    np.random.seed(2)
    random.seed(2)

    data_folder = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    all_subjects = ['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']
    str_aft = '_reduced_2fps_frame_index_withSegIndexAndIntKey.csv'
    input_types = ['img_crop']
    label_types = ['pain','segment_key']
    for horse in all_subjects:

        sampler = SegBasedSampler(data_folder, 
                     1200,
                     num_frames_per_seg = 240,
                     subjects = [horse],
                     randomize=False,
                     every_nth_segment=1,
                     str_aft = str_aft,
                     min_size = 10)
        pain_counts = np.array(sampler.get_pain_counts())
        pain_percent = pain_counts/np.sum(pain_counts)
        pain_all = list(pain_counts)+list(pain_percent)
        print (pain_all)
        str_print = '%d,%d,%.2f,%.2f'%tuple(pain_all)
        print (horse+','+str_print)
        # sampler.__iter__()
        # all_keys_check = []
        # for k in sampler.batch_keys:
        #     all_keys_check+=k
        # print (len(all_keys_check))
        # for k in all_keys_check:
        #     num_frames = len(sampler.frame_idx_dict[k])
        #     num_segs = num_frames//sampler.num_frames_per_seg
        #     rem = num_frames%sampler.num_frames_per_seg
        #     if rem>=sampler.min_size:
        #         num_segs += 1
        #     if num_segs!=all_keys_check.count(k):
        #         print (num_segs,all_keys_check.count(k),rem)

    
        # break


    # counts = [0,0]
    #     for key in self.all_keys:
    #         num_frames = len(self.frame_idx_dict[key])
    #         # num_frames = 1210
    #         num_segs = num_frames//self.num_frames_per_seg
    #         rem = num_frames%self.num_frames_per_seg
    #         if rem>self.min_size:
    #             num_segs += 1
    #         pain_label = self.label_dict.loc[self.label_dict['segment_key']==key,'pain'].iloc[0]
    #         counts[pain_label]+=num_segs

    #     return counts
    # sampler.__iter__()
    # print (len(sampler.batches), len(sampler.batch_keys))
    # for k in sampler.batch_keys:
    #     set_k = list(set(k))
    #     print (len(k),len(set_k))
    #     assert len(k)==len(set_k)
    # print (sampler.batch_keys[-1])
    # for k in sampler.batch_keys:
    #     print (k)
    



if __name__=='__main__':
    main()

