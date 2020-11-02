import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout

import random
import torch
import torch.nn as nn
import torch.autograd as A
import torch.nn.functional as F
import numpy as np

from rhodin.python.models import resnet_transfer
from rhodin.python.models import resnet_VNECT_3Donly

from rhodin.python.models.unet_utils import *
from rhodin.python.models import MLP
from models import unet_encode3D_clean
from . import pain_lstm_wbn
from helpers import util

raw_input = input

class PainHead(pain_lstm_wbn.PainHead):


    def augment_latent3d(self, rel_latent, rot_rel, rot_inv_rel):
        angles = np.linspace(0,2*np.pi,15)
        angle = np.random.choice(angles)
        rot_mat = util.rotationMatrixXZY(0, angle,0)
        rot_mat = torch.from_numpy(rot_mat).float().cuda()

        batch_size = rel_latent.size(0)
        rot_mat = rot_mat.unsqueeze(0).repeat(batch_size,1,1)
        cam2cam = torch.bmm(rot_rel,torch.bmm(rot_mat,rot_inv_rel))
        latent_3d = rel_latent.view(batch_size,-1,3)
        latent_3d_rotated = torch.bmm(latent_3d, cam2cam.transpose(1,2))
        latent_3d_rotated = latent_3d_rotated.view(batch_size,-1)
        return latent_3d_rotated




    def forward_pain(self, input_dict):
        # print (self.training)
        input = input_dict['img_crop']
        segment_key = input_dict['segment_key']
        batch_size = input_dict['img_crop'].size()[0]
        
        rot_inv = input_dict['extrinsic_rot_inv'].float()
        rot = input_dict['extrinsic_rot'].float()

        device = input.device
        
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat)
        
        h_n_all = []
        segment_key_new = []
        for segment_val in torch.unique_consecutive(segment_key):
            segment_bin = segment_key==segment_val
            
            rel_latent = latent_3d[segment_bin,:]
            rot_inv_rel = rot_inv[segment_bin,:]
            rot_rel = rot[segment_bin,:]
            # print (self.training)
            if self.training:
                rel_latent = self.augment_latent3d(rel_latent, rot_rel, rot_inv_rel)

            rel_latent = self.pad_input(rel_latent)
            _,(h_n,c_n) = self.lstm(rel_latent)
            h_n = h_n[-1]
            h_n_all.append(h_n)
            segment_key_rel = segment_key[segment_bin][:h_n.size(0)]
            segment_key_new.append(segment_key_rel)

        h_n = torch.cat(h_n_all,axis = 0)
        # print (h_n.size())
        # h_n = h_n[:1]
        # print (h_n.size())
        fake = False
        if h_n.size(0)==1:
            h_n = torch.cat([h_n,h_n],axis = 0)
            fake = True
        output_pain = self.to_pain[1](h_n)
        if fake:
            # print (output_pain)
            assert output_pain.size(0)==2
            output_pain = output_pain[:1,:]
            # print (output_pain)

        segment_key_new = torch.cat(segment_key_new, axis=0)
        
        pain_pred = torch.nn.functional.softmax(output_pain, dim = 1)

        ###############################################
        # Select the right output
        output_dict_all = {'pain': output_pain, 'pain_pred': pain_pred, 'segment_key':segment_key_new} 
        output_dict = {}
        # print (self.output_types)
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict    

