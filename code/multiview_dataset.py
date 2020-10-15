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

class MultiViewDataset(Dataset):
    """Multi-view surveillance dataset of horses in their box."""
    def __init__(self, data_folder, bg_folder,
                 input_types, label_types, subjects,rot_folder=None,
                 mean=(0.485, 0.456, 0.406),  #TODO update these to horse dataset.
                 stdDev= (0.229, 0.224, 0.225),
                 use_sequential_frames=0,
                 str_aft = None
                 ):
        """
        Args:
        data_folder: str,
        input_types: [str],
        label_types: [str]
        """

        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        self.lookup_viewpoint = pd.read_csv('../metadata/viewpoints.csv')

        class Image256toTensor(object):
            def __call__(self, pic):
                img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
                img = img.div(255)
                return img

            def __repr__(self):
                return self.__class__.__name__ + '()'

        self.transform_in = torchvision.transforms.Compose([
            Image256toTensor(), # torchvision.transforms.ToTensor() the torchvision one behaved differently for different pytorch versions, hence the custom one..
            torchvision.transforms.Normalize(self.mean, self.stdDev)
        ])
        self.label_dict = get_label_df_for_subjects(data_folder, subjects, str_aft = str_aft).to_dict()

    def __len__(self):
        return len(self.label_dict['frame'])

    def get_local_indices(self, index):
        input_dict = {}
        interval = self.label_dict['interval'][index]
        interval_ind = self.label_dict['interval_ind'][index]
        view = self.label_dict['view'][index]
        subject = self.label_dict['subject'][index]
        frame = self.label_dict['frame'][index]
        return interval, interval_ind, view, subject, frame

    def get_bg_path(self, view, subject):
            lookup_viewpoint = self.lookup_viewpoint.set_index('subject')
            camera = int(lookup_viewpoint.at[subject, str(view)])
            bg_path = self.bg_folder + 'median_0.1fps_camera_{}.jpg'.format(camera-1)
            return bg_path

    def get_rot_path(self, view, subject, key):
        lookup_viewpoint = self.lookup_viewpoint.set_index('subject')
        camera = int(lookup_viewpoint.at[subject, str(view)])
        rot_path = self.rot_folder + '{}_{}.npy'.format(key,camera)
        return rot_path            

    def load_image(self, name):
        return np.array(self.transform_in(imageio.imread(name)), dtype='float32')
    

    def __getitem__(self, index):
        interval, interval_ind, view, subject, frame = self.get_local_indices(index)

        def get_image_path(key):
            frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
                                str(view), '%06d'%frame])
            return self.data_folder + '/{}/{}/{}/{}.jpg'.format(subject,
                                                                interval,
                                                                view,
                                                                frame_id)
        def load_data(types):
            new_dict = {}
            for key in types:
                if key == 'img_crop':
                    new_dict[key] = self.load_image(get_image_path(key)) 
                elif key == 'bg_crop':
                    new_dict[key] = self.load_image(self.get_bg_path(view, subject))
                elif key == 'pain':
                    new_dict[key] = int(self.label_dict[key][index])
                elif key == 'view':
                    new_dict[key] = int(self.label_dict[key][index])
                elif key == 'segment_key':
                    new_dict[key] = self.label_dict[key][index]
                elif key == 'img_path':
                    interval_int = [int(val) for val in interval.split('_')]
                    new_dict[key] = np.array(interval_int+[interval_ind, view, frame])
                elif (key=='extrinsic_rot') or (key=='extrinsic_rot_inv') or (key=='extrinsic_tvec'):
                    rot_path = self.get_rot_path(view,subject,key)
                    new_dict[key] = np.load(rot_path)
                    # print (new_dict[key])
                elif key.startswith('extrinsic_rot_'):
                    key_split = key.split('_')
                    key_curr = '_'.join(key_split[:-1])
                    view_curr = int(key_split[-1])
                    rot_path = self.get_rot_path(view_curr,subject,key_curr)
                    new_dict[key] = np.load(rot_path)
                else:
                    new_dict[key] = np.array(self.label_dict[key][index], dtype='float32')
            return new_dict

        return load_data(self.input_types), load_data(self.label_types)


class MultiViewDatasetCrop(MultiViewDataset):
    """Multi-view surveillance dataset of horses in their box."""
    def __init__(self, data_folder, bg_folder,
                 input_types, label_types, subjects,rot_folder=None,
                 mean=(0.485, 0.456, 0.406),  #TODO update these to horse dataset.
                 stdDev= (0.229, 0.224, 0.225),
                 use_sequential_frames=0,
                 str_aft = None,
                 bg_post_pend = None
                 ):
        """
        Args:
        data_folder: str,
        input_types: [str],
        label_types: [str]
        """
        
        super().__init__(data_folder, bg_folder,
                 input_types, label_types, subjects,rot_folder,
                 mean,
                 stdDev,
                 use_sequential_frames,str_aft)
        if bg_post_pend is None:
            self.bg_post_pend = '_bg'
        else:
            self.bg_post_pend = bg_post_pend
        
    def __getitem__(self, index):

        interval, interval_ind, view, subject, frame = self.get_local_indices(index)

        def get_image_path(key):
            frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
                                str(view), '%06d'%frame])
            return self.data_folder + '/{}/{}/{}/{}.jpg'.format(subject,
                                                                interval,
                                                                view,
                                                                frame_id)
        def get_bg_path(key):
            frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
                                str(view), '%06d'%frame])
            str_format = '/{}/{}/{}'+self.bg_post_pend+'/{}.jpg'
            file_curr = self.data_folder + str_format.format(subject,
                                                                interval,
                                                                view,
                                                                frame_id)
            return file_curr 
            
       
        def load_data(types):
            new_dict = {}
            for key in types:
                if key == 'img_crop':
                    new_dict[key] = self.load_image(get_image_path(key)) 
                elif key == 'bg_crop':
                    new_dict[key] = self.load_image(get_bg_path(key))
                elif key in self.label_dict.keys():
                 # == 'pain':
                    new_dict[key] = int(self.label_dict[key][index])
                elif key == 'img_path':
                    interval_int = [int(val) for val in interval.split('_')]
                    new_dict[key] = np.array(interval_int+[interval_ind, view, frame])
                # elif key == 'view':
                #     new_dict[key] = int(self.label_dict[key][index])
                elif (key=='extrinsic_rot') or (key=='extrinsic_rot_inv') or (key=='extrinsic_tvec'):
                    rot_path = self.get_rot_path(view,subject,key)
                    new_dict[key] = np.load(rot_path)
                    # print (new_dict[key])
                else:
                    new_dict[key] = np.array(self.label_dict[key][index], dtype='float32')
            return new_dict

        return load_data(self.input_types), load_data(self.label_types)

class MultiViewDatasetSampler(Sampler):
    """ This sampler decides how to iterate over the indices in the dataset.
        Prepares batches of sub-batches, where a sub-batch contains
        indices corresponding to frames from different views at t,
        and indices corresponding to frames from different
        views at t', from the same interval."""

    def __init__(self, data_folder, save_path, mode,
                 batch_size,
                 subjects=None,
                 use_subject_batches=0, use_view_batches=0,
                 randomize=True,
                 use_sequential_frames=0,
                 every_nth_frame=1,
                 str_aft = None):
        # save function arguments
        
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        # build view/subject datastructure
        self.label_dict = get_label_df_for_subjects(data_folder, subjects, str_aft = str_aft).to_dict()
        print('Establishing sequence association. Available labels:', list(self.label_dict.keys()))
        all_keys = set()
        viewsets = {}
        interval_keys = {}
        data_length = len(self.label_dict['frame'])
        with tqdm(total=data_length) as pbar:
            for index in range(data_length):
                pbar.update(1)
                sub_i = self.label_dict['subject'][index]
                view_i = self.label_dict['view'][index]
                interval_i = self.label_dict['interval'][index]
                frame_i = self.label_dict['frame'][index]

                if subjects is not None and sub_i not in subjects:
                    continue
                # A key is an 'absolute moment in time', regardless of view.
                key = (sub_i, interval_i, frame_i)
                # Each key points to its different views in a viewset by storing their indices.
                # A viewset is a collection of indices of the frames corresponding to
                # that moment in time, i.e. that key.
                if key not in viewsets:
                    viewsets[key] = {}
                viewsets[key][view_i] = index

                # Only add if accumulated enough views
                if len(viewsets[key]) >= self.use_view_batches:
                    all_keys.add(key)

                    if interval_i not in interval_keys:
                        interval_keys[interval_i] = set()
                    interval_keys[interval_i].add(key)

        self.all_keys = list(all_keys)
        # to get rid of random element
        self.all_keys.sort()

        self.viewsets = viewsets
        self.interval_keys = {interval: list(keyset)
                              for interval, keyset in interval_keys.items()}

        print('\n', 'Done initializing entire dataset, before every_nth sampling.')
        print('Listed {} viewsets ({} frames), from {} sequences'.format(
                                            len(self.viewsets),
                                            len(self.all_keys)*self.use_view_batches,
                                            len(interval_keys)))

    def __len__(self):
        return len(self.label_dict['frame'])

    def __iter__(self):
        index_list = []
        print("Randomizing dataset (MultiViewDatasetSampler.__iter__)")
        s_time = time.time()
        ind_batched_str = 'indices_batched_{}.npy'.format(self.mode)
        indices_batched_path = os.path.join(self.save_path, ind_batched_str)
        if not os.path.isfile(indices_batched_path):
            print('No batched index was saved, creating one...')
            # Iterate over all keys, i.e. all 'moments in time'
            with tqdm(total=len(self.all_keys)//self.every_nth_frame) as pbar:
                for index in range(0,len(self.all_keys), self.every_nth_frame):
                    pbar.update(1)
                    key = self.all_keys[index]
                    # print ('key',key)
                    # s = input()
                    def get_view_subbatch(key):
                        """ Given a key (a moment in time),
                            return x indices for that key,
                            where x = self.use_view_batches."""
                        viewset = self.viewsets[key]
                        # example viewset: {0: 2562, 1: 7738, 2: 12912, 3: 18088}
                        viewset_keys = list(viewset.keys())
                        assert self.use_view_batches <= len(viewset_keys)
                        if self.randomize:
                            shuffle(viewset_keys)
                        if self.use_view_batches == 0:
                            view_subset_size = 99
                        else:
                            view_subset_size = self.use_view_batches
                        view_indices = [viewset[k] for k in viewset_keys[:view_subset_size]]
                        return view_indices

                    index_list = index_list + get_view_subbatch(key)
                    
                    if self.use_subject_batches:
                        # Add indices for other time, t', from the same interval
                        # to disentangle pose from appearance
                        interval_i = key[1]
                        potential_keys = self.interval_keys[interval_i]
                        nb_potential_keys = len(potential_keys)
                        key_t_prime = potential_keys[np.random.randint(nb_potential_keys)]
                        index_list = index_list + get_view_subbatch(key_t_prime)

            subject_batch_factor = 1 + int(self.use_subject_batches > 0) # either 1 or 2
            view_batch_factor = max(1, self.use_view_batches)
            # The following number should be equal to the number of new indices
            # added per iteration in the above loop, so we can group them accordingly. 
            sub_batch_size = view_batch_factor*subject_batch_factor
            # Check that the following holds so we don't split up sub-batches.
            assert self.batch_size % sub_batch_size == 0
            # Check that we can safely reshape index_list into sub-batches.
            assert len(index_list) % sub_batch_size == 0
            indices_batched = np.array(index_list).reshape([-1,sub_batch_size])
            
            if os.path.exists(os.path.split(indices_batched_path)[1]): 
            # if condition to keep the nn code working. 
            # nth is different for testing, so savepath doesn't exist
                np.save(indices_batched_path, indices_batched)
        else:
            indices_batched = np.load(indices_batched_path)
        e_time = time.time()
        print('\n', 'Time to create or load batched index: ', e_time - s_time)
        if self.randomize:
            # Randomizes the order of the sub-batches.
            indices_batched = np.random.permutation(indices_batched)
        indices_batched = indices_batched.reshape([-1])[:(indices_batched.size//self.batch_size)*self.batch_size] # drop last frames
        return iter(indices_batched.reshape([-1,self.batch_size]))


def get_label_df_for_subjects(data_folder, subjects, str_aft = None):
    subject_fi_dfs = []
    print('Iterating over frame indices per subject (.csv files)')
    
    if str_aft is None:
        str_aft = '_reduced_frame_index.csv'
    # print (data_folder)
    # if 'oft' in data_folder:
    #     thresh = float(os.path.split(data_folder)[0].split('_')[-2])
    #     str_aft = '_'.join(['','reduced','thresh','%.2f'%float(thresh),'frame','index'])+'.csv'
    # else:
    #     str_aft = '_reduced_frame_index.csv'

    for subject in subjects:
        csv_file = os.path.join(data_folder,subject + str_aft)

        subject_frame_index_dataframe = pd.read_csv(csv_file)
        # print (csv_file,len(subject_frame_index_dataframe))
        subject_fi_dfs.append(subject_frame_index_dataframe)
    frame_index_df = pd.concat(subject_fi_dfs, ignore_index=True)
    return frame_index_df


if __name__ == '__main__':
    config_dict_module = rhodin_utils_io.loadModule("configs/config_train_rotation_bl.py")
    config_dict = config_dict_module.config_dict
    print (config_dict['save_every'])
    train_subjects = ['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']
    config_dict['train_subjects'] = train_subjects
    dataset = MultiViewDataset(data_folder=config_dict['dataset_folder_train'],
                                   bg_folder=config_dict['bg_folder'],
                                   input_types=config_dict['input_types'],
                                   label_types=config_dict['label_types_train'],
                                   subjects = config_dict['train_subjects'],
                                   rot_folder = config_dict['rot_folder'])

    # batch_sampler = MultiViewDatasetSampler(
    #              data_folder=config_dict['data_dir_path'],
    #              use_subject_batches=1, use_view_batches=2,
    #              batch_size=8,
    #              randomize=True)
    
    batch_sampler = MultiViewDatasetSampler(data_folder=config_dict['dataset_folder_train'],
              subjects=config_dict['train_subjects'],
              use_subject_batches=config_dict['use_subject_batches'], use_view_batches=config_dict['use_view_batches'],
              batch_size=config_dict['batch_size_train'],
              randomize=True,
              every_nth_frame=config_dict['every_nth_frame'])
    
    # loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=False,
                                             # collate_fn=rhodin_utils_datasets.default_collate_with_string)

    trainloader = DataLoader(dataset, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=False,
                             collate_fn=rhodin_utils_datasets.default_collate_with_string)
    # import ipdb; ipdb.set_trace()

    data_iterator = iter(trainloader)
    input_dict, label_dict = next(data_iterator)
    print (input_dict.keys(), label_dict.keys())


    print('Number of frames in dataset: ', len(dataset))

    
