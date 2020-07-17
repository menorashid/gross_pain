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
    def __init__(self, config_dict_file, config_dict):
        super(IgniteTestNVS,self).__init__()
        
        # TODO add all input and output to get to config by default.

        data_loader = self.load_data_test(config_dict, ted.get_parameter_description(config_dict))
        model = self.load_network(config_dict)
        model = model.to(device)
        self.model = model
        self.data_loader = data_loader
        self.data_iterator = iter(data_loader)
        self.config_dict = config_dict
        self.mse = torch.nn.MSELoss(reduction = 'none')
        

        # self.task = task
        
        # if erc is None:
        #     self.erc = np.eye(3)
        # else:
        # self.erg = erg
        # self.view = view
        # self.rot = self.set_rotmat()
        # self.erc = self.set_erc()

    def get_view_rotmat(self, view):
        test_subject = self.config_dict['test_subjects']
        assert len(test_subject)==1
        test_subject = test_subject[0]
        rot_path = self.data_loader.dataset.get_rot_path(view, test_subject, 'extrinsic_rot')
        rot = np.load(rot_path)
        return rot
        

    def predict(self, input_dict, label_dict):
        self.model.eval()
        with torch.no_grad():
            input_dict_cuda, label_dict_cuda = rhodin_utils_datasets.nestedDictToDevice((input_dict, label_dict), device=device)
            output_dict_cuda = self.model(input_dict_cuda)
            output_dict = rhodin_utils_datasets.nestedDictToDevice(output_dict_cuda, device='cpu')
        return output_dict

    def set_view(self, input_dict, view):
        if view is not None:
            rot = self.get_view_rotmat(view)
            rot_curr = rot[np.newaxis,:,:]
            
            bs = input_dict['extrinsic_rot'].size(0)
            rot_curr = np.tile(rot_curr,(bs,1,1))
            input_dict['extrinsic_rot'] = torch.from_numpy(rot_curr).float().to(device)
            
        return input_dict

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
    def get_images(self, input_to_get = ['img_crop'], output_to_get = ['img_crop'], view = None):
        ret_vals = self.get_values(input_to_get, output_to_get, view)
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

    def get_latent_diff(self, views):
        # output_to_get = ['latent_3d']
        # input_to_get = ['view','frame']
        
        assert self.model.subbatch_size == 4
        input_keys = ['img_path','view']
        idx = 0
        all_diffs = []
        
        ret_vals = {}
        for key in ['diffs']+input_keys:
            ret_vals[key] = []


        for input_dict, label_dict in self.data_iterator:
            idx+=1
            assert 'view' in input_dict.keys()
            assert 'frame' in input_dict.keys()
            
            input_view = input_dict['view']
            input_frame = input_dict['frame']

            # print (input_view)
            # print (input_frame)
            # all_latents = []

            output_dict = self.predict( self.set_view(input_dict, view=None), label_dict)
            assert 'latent_3d' in output_dict.keys()

            gt_latent  = output_dict['latent_3d']

                
            diffs = []
            for view in views:
                input_dict = self.set_view(input_dict, view)
                output_dict = self.predict( input_dict, label_dict)
                latent = output_dict['latent_3d']
                rel_gt = gt_latent[input_view==view].unsqueeze(1)
                latent = latent.view(latent.size(0)//4,-1,latent.size(1),latent.size(2))
                diff = self.mse(rel_gt, latent)
                diff = torch.mean(torch.sqrt(torch.sum(diff, dim = -1)),-1)
                diff = diff.view(diff.size(0)*diff.size(1),1)
                diffs.append(diff)
                # print (diff[:10])
                # print (diff.size())

            diffs = torch.cat(diffs, dim = 1)
            diffs = torch.sum(diffs, dim = 1)/3.
            diffs = diffs.numpy()
            for key in input_keys:
                ret_vals[key].append(input_dict[key].numpy())
            ret_vals['diffs'].append(diffs)
            

        return ret_vals

                
    def get_values(self, input_to_get, output_to_get, view = None):
        
        the_rest = []
        for vals in [input_to_get,output_to_get]:
            dict_curr = {}
            for str_curr in vals:
                dict_curr[str_curr] = []
            the_rest.append(dict_curr)
        
        idx = 0
        
        for input_dict, label_dict in self.data_iterator:
            idx+=1

            input_dict = self.set_view(input_dict, view)
            
            output_dict = self.predict( input_dict, label_dict)

            dicts =[ input_dict, output_dict]
            for idx_dict,vals in enumerate([input_to_get, output_to_get]):
                
                for str_curr in vals:
                    the_rest[idx_dict][str_curr].append(dicts[idx_dict][str_curr].numpy())

            
        return the_rest
