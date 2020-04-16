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


class MultiViewDataset(Dataset):
    """Multi-view surveillance dataset of horses in their box."""
    def __init__(self, data_folder, bg_folder,
                 input_types, label_types, subjects,
                 mean=(0.485, 0.456, 0.406),  #TODO update these to horse dataset.
                 stdDev= (0.229, 0.224, 0.225),
                 use_sequential_frames=0,
                 ):
        """
        Args:
        data_folder: str,
        input_types: [str],
        label_types: [str]
        """

        for arg,val in list(locals().items()):
            setattr(self, arg, val)

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
        self.label_dict = get_label_dict(data_folder, subjects)

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

    def __getitem__(self, index):

        interval, interval_ind, view, subject, frame = self.get_local_indices(index)

        def get_image_path(key):
            frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
                                str(view), '%06d'%frame])
            return self.data_folder + '/{}/{}/{}/{}.jpg'.format(subject,
                                                                interval,
                                                                view,
                                                                frame_id)
        def get_bg_path(view, subject):
            lookup_viewpoint = pd.read_csv('../metadata/viewpoints.csv').set_index('subject')
            camera = int(lookup_viewpoint.at[subject, str(view)])
            bg_path = self.bg_folder + 'median_0.1fps_camera_{}.jpg'.format(camera-1)
            return bg_path

        def load_image(name):
            return np.array(self.transform_in(imageio.imread(name)), dtype='float32')
        
        def load_data(types):
            new_dict = {}
            for key in types:
                print (key)
                if key == 'img_crop':
                    new_dict[key] = load_image(get_image_path(key)) 
                elif key == 'bg_crop':
                    new_dict[key] = load_image(get_bg_path(view, subject))
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

    def __init__(self, data_folder, batch_size,
                 subjects=None,
                 use_subject_batches=0, use_view_batches=0,
                 randomize=True,
                 use_sequential_frames=0,
                 every_nth_frame=1):
        # save function arguments
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        # build view/subject datastructure
        self.label_dict = get_label_dict(data_folder, subjects)
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
                # Each key points to its different views by storing their indices.
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
        self.viewsets = viewsets
        self.interval_keys = {interval: list(keyset)
                              for interval, keyset in interval_keys.items()}

        print("DictDataset: Done initializing, listed {} viewsets ({} frames) and {} sequences".format(
                                            len(self.viewsets), len(self.all_keys), len(interval_keys)))

    def __len__(self):
        return len(self.label_dict['frame'])

    def __iter__(self):
        index_list = []
        print("Randomizing dataset (MultiViewDatasetSampler.__iter__)")
        # Iterate over all keys, i.e. all 'moments in time'
        with tqdm(total=len(self.all_keys)//self.every_nth_frame) as pbar:
            for index in range(0,len(self.all_keys), self.every_nth_frame):
                pbar.update(1)
                key = self.all_keys[index]
                def get_view_subbatch(key):
                    """ Given a key (a moment in time),
                        return x indices for that key,
                        where x = self.use_view_batches."""
                    viewset = self.viewsets[key]
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
                    # Add indices for random moment (t') from the same interval.
                    # These indices can be from any viewpoint.
                    # I suspect that this is the appearance branch.
                    interval_i = key[1]
                    potential_keys = self.interval_keys[interval_i]
                    key_other = potential_keys[np.random.randint(len(potential_keys))]
                    index_list = index_list + get_view_subbatch(key_other)

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
        if self.randomize:
            # Randomizes the order of the sub-batches.
            indices_batched = np.random.permutation(indices_batched)
        indices_batched = indices_batched.reshape([-1])[:(indices_batched.size//self.batch_size)*self.batch_size] # drop last frames
        return iter(indices_batched.reshape([-1,self.batch_size]))


def get_label_dict(data_folder, subjects):
    subject_fi_dfs = []
    print('Iterating over frame indices per subject (.csv files)')
    for subject in subjects:
        subject_frame_index_dataframe = pd.read_csv(data_folder + subject + '_reduced_frame_index.csv')
        subject_fi_dfs.append(subject_frame_index_dataframe)
    frame_index_df = pd.concat(subject_fi_dfs, ignore_index=True)
    label_dict = frame_index_df.to_dict()
    return label_dict


if __name__ == '__main__':
    config_dict_module = rhodin_utils_io.loadModule("configs/config_train.py")
    config_dict = config_dict_module.config_dict

    dataset = MultiViewDataset(
                 data_folder=config_dict['data_dir_path'],
                 input_types=['img_crop'], label_types=['img_crop'])

    batch_sampler = MultiViewDatasetSampler(
                 data_folder=config_dict['data_dir_path'],
                 use_subject_batches=1, use_view_batches=2,
                 batch_size=8,
                 randomize=True)

    trainloader = DataLoader(dataset, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=False,
                             collate_fn=rhodin_utils_datasets.default_collate_with_string)
    import ipdb; ipdb.set_trace()

    data_iterator = iter(trainloader)
    input_dict, label_dict = next(data_iterator)


    print('Number of frames in dataset: ', len(dataset))

    
