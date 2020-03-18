import pandas as pd
import numpy as np
import torchvision
import imageio
import torch
import os

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from helpers import util
from tqdm import tqdm


class MultiViewFrameDataset(Dataset):
    """Multi-view surveillance dataset of horses in their box."""
    def __init__(self, data_folder, 
                 input_types, label_types,
                 mean=(0.485, 0.456, 0.406),
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
        label_dict = pd.read_csv(data_folder + '/labels.csv').to_dict()
        print('Loading .csv label file to memory')
        self.label_dict = label_dict

    def __len__(self):
        return len(self.label_dict['frame'])

    def get_local_indices(self, index):
        input_dict = {}
        cam = int(self.label_dict['cam'][index])
        seq = int(self.label_dict['seq'][index])
        frame = int(self.label_dict['frame'][index])
        subject = int(self.label_dict['subject'][index])
        return cam, seq, frame, subject

    def __getitem__(self, index):

        cam, seq, frame, subject = self.get_local_indices(index)

        def get_image_name(key):
            id_str = 'as' if subject == 0 else 'br'
            frame_id = '_'.join(id_str, str(f'{seq:02}'), str(cam), str(f'{frame:06}'))
            frame_id += '_'
            return self.data_folder + '/subj_{}/seq_{:02d}/{}/{}_{:06d}.jpg'.format(subject,
                                                                                    seq,
                                                                                    cam,
                                                                                    frame_id,
                                                                                    frame)
        def load_image(name):
            return np.array(self.transform_in(imageio.imread(name)), dtype='float32')

        def load_data(types):
            new_dict = {}
            for key in types:
                if key in ['img_crop','bg_crop']:
                    new_dict[key] = load_image(get_image_name(key)) 
                else:
                    new_dict[key] = np.array(self.label_dict[key][index], dtype='float32')
            return new_dict

        return load_data(self.input_types), load_data(self.label_types)


class MultiViewFrameDatasetSampler(Sampler):
    def __init__(self, data_folder, batch_size,
                 horse_subset=None,
                 use_subject_batches=0, use_cam_batches=0,
                 randomize=True,
                 use_sequential_frames=0,
                 every_nth_frame=1):
        # save function arguments
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        # build cam/subject datastructure
        label_dict = pd.read_csv(data_folder + '/labels.csv').to_dict()
        print('Loading .csv label file to memory')
        self.label_dict = label_dict
        print('Establishing sequence association. Available labels:', list(label_dict.keys()))
        all_keys = set()
        camsets = {}
        sequence_keys = {}
        data_length = len(label_dict['frame'])
        with tqdm(total=data_length) as pbar:
            for index in range(data_length):
                pbar.update(1)
                sub_i = label_dict['subject'][index]
                cam_i = label_dict['cam'][index]
                seq_i = label_dict['seq'][index]
                frame_i = label_dict['frame'][index]

                if horse_subset is not None and sub_i not in horse_subset:
                    continue

                key = (sub_i, seq_i, frame_i)
                if key not in camsets:
                    camsets[key] = {}
                camsets[key][cam_i] = index

                # only add if accumulated enough cameras
                if len(camsets[key]) >= self.use_cam_batches:
                    all_keys.add(key)

                    if seq_i not in sequence_keys:
                        sequence_keys[seq_i] = set()
                    sequence_keys[seq_i].add(key)

        self.all_keys = list(all_keys)
        self.camsets = camsets
        self.sequence_keys = {seq: list(keyset) for seq, keyset in sequence_keys.items()}
        print("DictDataset: Done initializing, listed {} camsets ({} frames) and {} sequences".format(
                                            len(self.camsets), len(self.all_keys), len(sequence_keys)))

    def __iter__(self):
        index_list = []
        print("Randomizing dataset (MultiViewFrameDatasetSampler.__iter__)")
        with tqdm(total=len(self.all_keys)//self.every_nth_frame) as pbar:
            for index in range(0,len(self.all_keys), self.every_nth_frame):
                pbar.update(1)
                key = self.all_keys[index]
                def get_cam_subbatch(key):
                    camset = self.camsets[key]
                    cam_keys = list(camset.keys())
                    assert self.use_cam_batches <= len(cam_keys)
                    if self.randomize:
                        shuffle(cam_keys)
                    if self.use_cam_batches == 0:
                        cam_subset_size = 99
                    else:
                        cam_subset_size = self.use_cam_batches
                    cam_indices = [camset[k] for k in cam_keys[:cam_subset_size]]
                    return cam_indices

                index_list = index_list + get_cam_subbatch(key)
                if self.use_subject_batches:
                    seqi = key[1]
                    potential_keys = self.sequence_keys[seqi]
                    key_other = potential_keys[np.random.randint(len(potential_keys))]
                    index_list = index_list + get_cam_subbatch(key_other)

        subject_batch_factor = 1+int(self.use_subject_batches > 0) # either 1 or 2
        cam_batch_factor = max(1,self.use_cam_batches)
        sub_batch_size = cam_batch_factor*subject_batch_factor
        assert len(index_list) % sub_batch_size == 0
        indices_batched = np.array(index_list).reshape([-1,sub_batch_size])
        if self.randomize:
            indices_batched = np.random.permutation(indices_batched)
        indices_batched = indices_batched.reshape([-1])[:(indices_batched.size//self.batch_size)*self.batch_size] # drop last frames
        return iter(indices_batched.reshape([-1,self.batch_size]))


if __name__ == '__main__':
    config_dict_module = util.load_module("configs/config_train.py")
    config_dict = config_dict_module.config_dict

    dataset = MultiViewFrameDataset(
                 data_folder=config_dict['data_dir_path'],
                 input_types=['img_crop'], label_types=['img_crop'])

    batch_sampler = MultiViewFrameDatasetSampler(
                 data_folder=config_dict['data_dir_path'],
                 use_subject_batches=1, use_cam_batches=2,
                 batch_size=8,
                 randomize=True)

    trainloader = DataLoader(dataset, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=False,
                             collate_fn=util.default_collate_with_string)
    
