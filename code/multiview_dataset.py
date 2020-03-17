import pandas as pd
import numpy as np
import imageio
import torch
import os

from torch.utils.data import Dataset, DataLoader
from helpers import util


class MultiViewFrameDataset(Dataset):
    """Multi-view surveillance dataset of horses in their box."""

    def __init__(self, config_dict):
        """
        Args:
        config_dict: {} config dictionary.
        """
        self.config_dict = config_dict
        import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.label_dict['frame'])

    def get_local_indices(self, index):
        input_dict = {}
        cam = int(self.label_dict['cam'][index].item())
        seq = int(self.label_dict['seq'][index].item())
        frame = int(self.label_dict['frame'][index].item())
        return cam, seq, frame

    def __getitem__(self, index):
        cam, seq, frame = self.get_local_indices(index)
        def get_image_name(key):
            return self.data_folder + '/seq_{:03d}/cam_{:02d}/{}_{:06d}.png'.format(seq,
                                                                                    cam,
                                                                                    key,
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

if __name__ == '__main__':
    config_dict_module = util.load_module("configs/config_train.py")
    config_dict = config_dict_module.config_dict

    dataset = MultiViewFrameDataset(config_dict)
    
