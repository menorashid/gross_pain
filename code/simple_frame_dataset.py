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


class SimpleFrameDataset(Dataset):
    """Single frame surveillance dataset of horses in their box."""
    def __init__(self, data_folder, 
                 subjects, input_types, label_types,
                 mean=(0.485, 0.456, 0.406),  #TODO update these to horse dataset.
                 stdDev= (0.229, 0.224, 0.225),
                 ):
        """
        Args:
        data_folder: str,
        subjects: [str]
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
        self.label_dict = get_label_df_for_subjects(data_folder, subjects).to_dict()

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
        def load_image(name):
            return np.array(self.transform_in(imageio.imread(name)), dtype='float32')
        
        def load_data(input_types):
            for key in input_types:
                if key == 'img_crop':
                    data = load_image(get_image_path(key)) 
                if key == 'pain':
                    data = int(self.label_dict[key][index])
                if key == 'view':
                    data = int(self.label_dict[key][index])
            return data

        return load_data(self.input_types), load_data(self.label_types)


class SimpleRandomFrameSampler(Sampler):
    def __init__(self, data_folder, 
                 subjects=None,
                 views=None,
                 every_nth_frame=1):
        # Reduce frame index to wanted subjects and views
        subject_label_df = get_label_df_for_subjects(data_folder, subjects)
        subject_view_label_df = only_keep_wanted_views(subject_label_df, views)

        # Get the wanted proportion of the data
        subject_view_label_df = subject_view_label_df.sample(frac=1/every_nth_frame)

        self.sample_index = subject_view_label_df.index.tolist()

        self.label_dict = subject_view_label_df.to_dict()

    def __len__(self):
        return len(self.label_dict['frame'])

    def __iter__(self):
        return iter(self.sample_index)


def only_keep_wanted_views(label_df, views):
    all_views = [0,1,2,3]
    if set(views) == set(all_views):
        return label_df
    else:
        indices_to_drop = []
        unwanted_views = set(all_views) - set(views)
        for ind, row in label_df:
            view = row['view']
            if view in unwanted_views:
                indices_to_drop.append(ind)
        df_reduced = label_df.drop((label_df.index[indices_to_drop]))
        df_reduced.reset_index(drop=True)
        return df_reduced


def get_label_df_for_subjects(data_folder, subjects):
    subject_fi_dfs = []
    print('Iterating over frame indices per subject (.csv files)')
    for subject in subjects:
        subject_frame_index_dataframe = pd.read_csv(data_folder + subject + '_reduced_frame_index.csv')
        subject_fi_dfs.append(subject_frame_index_dataframe)
    frame_index_df = pd.concat(subject_fi_dfs, ignore_index=True)
    return frame_index_df


if __name__ == '__main__':
    config_dict_module = rhodin_utils_io.loadModule("configs/config_pain_debug.py")
    config_dict = config_dict_module.config_dict
    print (config_dict['save_every'])
    train_subjects = ['aslan','brava','herrera','julia','kastanjett','naughty_but_nice','sir_holger']
    config_dict['train_subjects'] = train_subjects
    config_dict['dataset_folder_train'] = '/local_storage/users/sbroome/SLU_LPS/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
    dataset = SimpleFrameDataset(data_folder=config_dict['dataset_folder_train'],
                                 subjects = config_dict['train_subjects'],
                                 input_types=config_dict['input_types'],
                                 label_types=config_dict['label_types_train'])

    sampler = SimpleRandomFrameSampler(
              data_folder=config_dict['dataset_folder_train'],
              subjects=config_dict['train_subjects'],
              views=config_dict['views'],
              every_nth_frame=config_dict['every_nth_frame'])
    
    trainloader = DataLoader(dataset, sampler=sampler,
                             batch_size=config_dict['batch_size_train'],
                             num_workers=0, pin_memory=False,
                             collate_fn=rhodin_utils_datasets.default_collate_with_string)

    data_iterator = iter(trainloader)
    input_batch, label_batch = next(data_iterator)
    print (input_batch.shape, label_batch.shape)


    print('Number of frames in dataset: ', len(dataset))

