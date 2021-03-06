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

# class SegBasedDataset(MultiViewDataset):
#     """Multi-view surveillance dataset of horses in their box."""
#     def __init__(self, data_folder,
#                  input_types, label_types, subjects,
#                  mean=(0.485, 0.456, 0.406),  #TODO update these to horse dataset.
#                  stdDev= (0.229, 0.224, 0.225),
#                  str_aft = None
#                  ):
#         """
#         Args:
#         data_folder: str,
#         input_types: [str],
#         label_types: [str]
#         """
        
#         super().__init__(data_folder, data_folder,
#                  input_types, label_types, subjects,rot_folder = None,
#                  mean = mean,
#                  stdDev = stdDev,
#                  use_sequential_frames = 0,str_aft)
#         if bg_post_pend is None:
#             self.bg_post_pend = '_bg'
#         else:
#             self.bg_post_pend = bg_post_pend
        
#     def __getitem__(self, index):

#         interval, interval_ind, view, subject, frame = self.get_local_indices(index)

#         def get_image_path(key):
#             frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
#                                 str(view), '%06d'%frame])
#             return self.data_folder + '/{}/{}/{}/{}.jpg'.format(subject,
#                                                                 interval,
#                                                                 view,
#                                                                 frame_id)
#         def get_bg_path(key):
#             frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
#                                 str(view), '%06d'%frame])
#             str_format = '/{}/{}/{}'+self.bg_post_pend+'/{}.jpg'
#             file_curr = self.data_folder + str_format.format(subject,
#                                                                 interval,
#                                                                 view,
#                                                                 frame_id)
#             return file_curr 
            
       
#         def load_data(types):
#             new_dict = {}
#             for key in types:
#                 if key == 'img_crop':
#                     new_dict[key] = self.load_image(get_image_path(key)) 
#                 elif key == 'bg_crop':
#                     new_dict[key] = self.load_image(get_bg_path(key))
#                 elif key in self.label_dict.keys():
#                  # == 'pain':
#                     new_dict[key] = int(self.label_dict[key][index])
#                 elif key == 'img_path':
#                     interval_int = [int(val) for val in interval.split('_')]
#                     new_dict[key] = np.array(interval_int+[interval_ind, view, frame])
#                 # elif key == 'view':
#                 #     new_dict[key] = int(self.label_dict[key][index])
#                 elif (key=='extrinsic_rot') or (key=='extrinsic_rot_inv') or (key=='extrinsic_tvec'):
#                     rot_path = self.get_rot_path(view,subject,key)
#                     new_dict[key] = np.load(rot_path)
#                     # print (new_dict[key])
#                 else:
#                     new_dict[key] = np.array(self.label_dict[key][index], dtype='float32')
#             return new_dict

#         return load_data(self.input_types), load_data(self.label_types)



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
            if rem>self.min_size:
                num_segs += 1
            pain_label = self.label_dict.loc[self.label_dict['segment_key']==key,'pain'].iloc[0]
            counts[pain_label]+=num_segs

        return counts

    def get_pain_weight(self):
        train_counts = np.array(self.get_pain_counts())
        weights = 1/train_counts
        weights = weights/np.sum(weights) * 2
        return weights



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
        
        if (self.batches is None) or (self.randomize):
            index_list = []
            # s_time = time.time()
           
            batches = []
            curr_batch = []
            print("Making dataset SegBasedSampler")
            if self.randomize:
                print("Randomizing dataset (SegBasedSampler.__iter__)")
                random.shuffle(self.all_keys)

            for index in range(len(self.all_keys)):
                
                key = self.all_keys[index]
                frame_idx = self.frame_idx_dict[key]
                frame_idx.sort()
                
                num_frames = len(frame_idx)

                if num_frames>self.num_frames_per_seg:
                    new_batches = [frame_idx[x:x+self.num_frames_per_seg] for x in range(0,num_frames,self.num_frames_per_seg)]
                    
                    if len(new_batches[-1])<self.min_size:
                        new_batches = new_batches[:-1]
                    
                    batches, curr_batch = self.add_segments_to_batch(new_batches[::-1],batches, curr_batch)
                else:
                    batches, curr_batch = self.add_segments_to_batch([frame_idx], batches, curr_batch)

            self.batches = batches

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
    # dataset = MultiViewDataset(data_folder=data_folder,
    #                                    bg_folder=data_folder,
    #                                    input_types=input_types,
    #                                    label_types=label_types,
    #                                    subjects=all_subjects,
    #                                    rot_folder = None,
    #                                    str_aft = str_aft)
    counts = []
    for horse_id in all_subjects:
        sampler = SegBasedSampler(data_folder, 
                     1200,
                     num_frames_per_seg = 240,
                     subjects = [horse_id],
                     randomize=True,
                     every_nth_segment=1,
                     str_aft = str_aft,
                     min_size = 10)

        counts.append(sampler.get_pain_counts())
        # print (horse_id, counts)

    for idx_horse_id, horse_id in enumerate(all_subjects):
        test_count = np.array(counts[idx_horse_id])
        train_counts = [counts[idx] for idx in range(len(counts)) if idx is not idx_horse_id]
        train_counts = np.sum(np.array(train_counts),axis = 0)
        # train_counts = train_counts/np.sum(train_counts)
        # test_count = test_count/np.sum(test_count)
        print (horse_id)
        print (train_counts)
        print (train_counts/np.sum(train_counts))
        print (test_count/np.sum(test_count))

        weights = 1/train_counts
        weights = weights/np.sum(weights) * 2

        
        print ('online',weights)
        weights = 1/(train_counts/np.sum(train_counts))
        weights = weights/np.sum(weights)
        print ('me',weights)
        # # print (sampler.all_keys[:10])
        # vals = sampler.label_dict[['pain','segment_key']].drop_duplicates().values
        # bin_keep = np.in1d(vals[:,1],sampler.all_keys)
        # vals_kept = vals[bin_keep,0]
        # vals_zero = np.sum(vals_kept==0)
        # total = float(vals_kept.size)
        # print (horse_id, vals_zero/total,(total - vals_zero)/total, total)


        

    # print (vals[:10])

    # loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False,
    #                                          collate_fn=rhodin_utils_datasets.default_collate_with_string)

    # for batch in loader:
    #     # print (len(batch),type(batch[0]))
    #     # print(len(batch[0][0]))
    #     # print (batch[0][0]['img_crop'].shape)
    #     print (batch[0].keys())
    #     print (type(batch[0]['img_crop']), len(batch[0]['img_crop']))
    #     print (batch[0]['img_crop'].size())
    #     print (len(batch[1]['segment_key']))
    #     print (torch.unique(batch[1]['segment_key']))
    #     # print (batch.keys())
    #     s = input()




if __name__=='__main__':
    main()

