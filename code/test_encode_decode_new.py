import os
from helpers import util, visualize
import numpy as np
import pandas as pd
import numpy as np
import imageio
import torch
import sklearn.manifold
import sklearn.preprocessing

import train_encode_decode as ted
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import io as rhodin_utils_io

from train_encode_decode_pain import get_model_path 
import glob
# as ext_get_model_path

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

class IgniteTestNVS(ted.IgniteTrainNVS):
    def __init__(self, config_dict_file, config_dict, task):
        super(IgniteTestNVS,self).__init__()
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
        
        data_loader = self.load_data_test(config_dict, ted.get_parameter_description(config_dict))
        model = self.load_network(config_dict)
        model = model.to(device)
        self.model = model
        self.data_loader = data_loader
        self.data_iterator = iter(data_loader)
        self.config_dict = config_dict
        self.task = task

        # return model, data_loader, config_dict
    def predict(self, input_dict, label_dict):
        self.model.eval()
        with torch.no_grad():
            input_dict_cuda, label_dict_cuda = rhodin_utils_datasets.nestedDictToDevice((input_dict, label_dict), device=device)
            output_dict_cuda = self.model(input_dict_cuda)
            output_dict = rhodin_utils_datasets.nestedDictToDevice(output_dict_cuda, device='cpu')
        return output_dict

    def nextImage(self):
        input_dict, label_dict = next(self.data_iterator)
        if self.task.startswith('simple'):
            input_dict['external_rotation_cam'] = torch.from_numpy(np.eye(3)).float().to(device)
            input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().to(device)
        return input_dict, label_dict

    # get im mse loss
    # get rotated loss

    def unnormalize(self, batch):
        vals = []
        for val in ['img_mean', 'img_std']:
            val = self.config_dict[val]
            val = np.array(val)[np.newaxis,:,np.newaxis,np.newaxis]
            vals.append(val)

        batch = batch *vals[1] +vals[0]
        batch = np.uint8(np.clip(batch,0.,1.)*255)
        return batch
        

    # get images
    def get_images(self, input_to_get = ['img_crop'], output_to_get = ['img_crop']):
        ret_vals = self.get_values(input_to_get, output_to_get)
        keys = [input_to_get, output_to_get]
        mean = self.config_dict['img_mean']
        stdDev = self.config_dict['img_std']
        for idx_ret_val, ret_val in enumerate(ret_vals):
            key_arr = keys[idx_ret_val]
            for key_val in key_arr:
                for idx_batch, batch in enumerate(ret_val[key_val]):
                    if 'crop' in key_val:
                        batch_un_norm = self.unnormalize(batch)
                    
                    batch_un_norm = np.swapaxes(np.swapaxes(batch_un_norm, 1, 3),1,2)
                    ret_val[key_val][idx_batch] = batch_un_norm

        return ret_vals


    def get_values(self, input_to_get, output_to_get):
        
        the_rest = []
        for vals in [input_to_get,output_to_get]:
            dict_curr = {}
            for str_curr in vals:
                dict_curr[str_curr] = []
            the_rest.append(dict_curr)
        
        idx = 0
        for input_dict, label_dict in self.data_iterator:
            idx+=1
            # input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().to(device)
            output_dict = self.predict( input_dict, label_dict)

            dicts =[ input_dict, output_dict]
            for idx_dict,vals in enumerate([input_to_get, output_to_get]):
                print (dicts[idx_dict].keys(),the_rest[idx_dict])
                for str_curr in vals:
                    the_rest[idx_dict][str_curr].append(dicts[idx_dict][str_curr].numpy())

            # for str_curr in output_to_get:
            #     the_rest[str_curr].append(output_dict[str_curr].numpy())

        return the_rest
