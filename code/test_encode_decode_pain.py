import os
from helpers import util, visualize
import numpy as np
import pandas as pd
import numpy as np
import imageio
import torch
import sklearn.manifold
import sklearn.preprocessing

import train_encode_decode_pain as tedp
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import io as rhodin_utils_io

from test_encode_decode_new import IgniteTestNVS
from train_encode_decode_pain import get_model_path 
import glob
from tqdm import tqdm

# as ext_get_model_path

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

class IgniteTestPain(tedp.IgniteTrainPainFromLatent, IgniteTestNVS):
    def __init__(self, config_dict_file, config_dict):
        data_loader = self.load_data_test(config_dict, tedp.get_parameter_description_pain(config_dict))
        model = self.load_network(config_dict)
        model = model.to(device)
        self.model = model
        self.data_loader = data_loader
        self.data_iterator = iter(data_loader)
        self.config_dict = config_dict


    def get_accuracy(self):
        input_to_get = ['pain']
        output_to_get = ['pain_pred']

        values = self.get_values( input_to_get, output_to_get, view = None, bg = None)
        gt = np.concatenate(values[0]['pain'])
        pred = np.argmax(np.concatenate(values[1]['pain_pred']), axis = 1)
        accuracy = np.sum(pred==gt)/gt.size
        print (accuracy, gt.size)
        # print (pred)
        # print (gt)
        # print (len(values[0]['pain']),values[0]['pain'][0].shape,values[0]['pain'][0])
        # print (len(values[1]['pain']),values[1]['pain'][0].shape)

        return accuracy
