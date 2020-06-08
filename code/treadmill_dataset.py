import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scipy.io as scio
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


class TreadmillDataset(Dataset):
    """Single frame surveillance dataset of horses in their box."""
    def __init__(self, mocap_folder, rgb_folder, 
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

        class ResizeTensor(object):
            def __call__(self, img_tensor):
                img_tensor_batch = torch.unsqueeze(img_tensor, 0)
                img_batch = torch.nn.functional.interpolate(img_tensor_batch, size=(128,128))
                img = torch.squeeze(img_batch)
                return img

            def __repr__(self):
                return self.__class__.__name__ + '()'

        self.transform_in = torchvision.transforms.Compose([
            Image256toTensor(), # torchvision.transforms.ToTensor() the torchvision one behaved differently for different pytorch versions, hence the custom one..
            ResizeTensor(),
            torchvision.transforms.Normalize(self.mean, self.stdDev)
        ])
        self.label_dict = get_label_df_for_subjects(subjects).to_dict()

    def __len__(self):
        return len(self.label_dict['rgb_index'])

    def get_local_indices(self, index):
        input_dict = {}
        clip_id = self.label_dict['clip_id'][index]
        mocap_ind = self.label_dict['mocap_index'][index]
        rgb_ind = self.label_dict['rgb_index'][index]
        subject = clip_id[:3]
        return clip_id, mocap_ind, rgb_ind, subject

    def __getitem__(self, index):

        clip_id, mocap_ind, rgb_ind, subject = self.get_local_indices(index)

        def get_image_path(key):
            frame_index =  '%04d'%rgb_ind
            return self.rgb_folder + '/{}/{}_01/frame{}.png'.format(clip_id,
                                                               clip_id,
                                                               frame_index)
        def get_mocap_path(key):
            return self.mocap_folder + '/{}.mat'.format(clip_id)

        def load_image(path):
            image = np.array(self.transform_in(imageio.imread(path)), dtype='float32')
            print(image.shape)
            return image

        def load_pose(path):
            nested_mocap = scio.loadmat(path)
            # Mocap XYZ with residual
            mocap_4d = nested_mocap[clip_id]['Trajectories'][0][0][0][0][0][0]['Data'][0]
            mocap_3d = mocap_4d[:,:3,:]  # Just mocap
            # Some are 50, some are 51 long
            if mocap_3d.shape[0] == 51:  # Remove the first mocap joint "CristaFac_L"
                mocap_3d = mocap_3d[1:,:,:]
            return mocap_3d
        
        def load_data(input_types):
            new_dict = {}
            for key in input_types:
                if key == 'img_crop':
                    new_dict[key] = load_image(get_image_path(key)) 
                if key == 'pose':
                    new_dict[key] = load_pose(get_mocap_path(key))
            return new_dict

        return load_data(self.input_types), load_data(self.label_types)


class TreadmillRandomFrameSampler(Sampler):
    def __init__(self,
                 subjects=None,
                 every_nth_frame=1):
        # Reduce frame index to wanted subjects and views
        subject_label_df = get_label_df_for_subjects(subjects)

        # Get the wanted proportion of the data. This shuffles the rows of the df.
        subject_label_df = subject_label_df.sample(frac=1/every_nth_frame)

        self.sample_index = subject_label_df.index.tolist()

        self.label_dict = subject_label_df.to_dict()

    def __len__(self):
        return len(self.label_dict['frame'])

    def __iter__(self):
        return iter(self.sample_index)


def get_label_df_for_subjects(subjects):
    subject_fi_dfs = []
    print('Iterating over frame indices per subject (.csv files)')
    for subject in subjects:
        subject_frame_index_dataframe = pd.read_csv('../metadata/treadmill_frame_index_'  + subject + '.csv')
        subject_fi_dfs.append(subject_frame_index_dataframe)
    frame_index_df = pd.concat(subject_fi_dfs, ignore_index=True)
    return frame_index_df


if __name__ == '__main__':
    config_dict_module = rhodin_utils_io.loadModule("configs/config_pose_debug.py")
    config_dict = config_dict_module.config_dict
    print (config_dict['save_every'])
    train_subjects = ['ART', 'HOR', 'LAC', 'LAR', 'LAZ', 'LEA', 'LOR', 'PRA']
    config_dict['train_subjects'] = train_subjects
    config_dict['dataset_folder_train'] = '/Midgard/home/sbroome/lameness/'
    config_dict['dataset_folder_train_mocap'] = '/Midgard/home/sbroome/lameness/treadmill_lameness_mocap_ci_may11/mocap/'
    config_dict['dataset_folder_train_rgb'] = '/Midgard/home/sbroome/lameness/animals_data/'
    dataset = TreadmillDataset(rgb_folder=config_dict['dataset_folder_train_rgb'],
                               mocap_folder=config_dict['dataset_folder_train_mocap'],
                               subjects = config_dict['train_subjects'],
                               input_types=config_dict['input_types'],
                               label_types=config_dict['label_types_train'])

    sampler = SimpleRandomFrameSampler(
              subjects=config_dict['train_subjects'],
              every_nth_frame=config_dict['every_nth_frame'])
    
    trainloader = DataLoader(dataset, sampler=sampler,
                             batch_size=config_dict['batch_size_train'],
                             num_workers=0, pin_memory=False,
                             collate_fn=rhodin_utils_datasets.default_collate_with_string)

    data_iterator = iter(trainloader)
    input_batch, label_batch = next(data_iterator)
    print (input_batch.shape, label_batch.shape)


    print('Number of frames in dataset: ', len(dataset))

