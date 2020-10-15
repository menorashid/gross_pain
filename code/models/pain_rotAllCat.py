import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout

import random
import torch
import torch.autograd as A
import torch.nn.functional as F
import numpy as np

from rhodin.python.models import resnet_transfer
from rhodin.python.models import resnet_VNECT_3Donly

from rhodin.python.models.unet_utils import *
from rhodin.python.models import MLP
from models.pain_simple import PainHead as PainSimple
import re
raw_input = input

class PainHead(nn.Module):

    def __init__(self, base_network , output_types, n_hidden_to_pain = 2, d_hidden = 2048, dropout = 0.5 ,num_views = 4):
        super(PainHead, self).__init__()
        self.encoder = base_network.encoder
        self.to_3d = base_network.to_3d
        self.num_views = num_views
        self.to_pain = MLP.MLP_fromLatent(d_in = 4*base_network.dimension_3d, d_hidden=d_hidden, d_out=2, n_hidden=n_hidden_to_pain, dropout=dropout)
        self.output_types = output_types

    def get_latent_concatenated(self,latent_3d, input_dict):
        batch_size = input_dict['img_crop'].size()[0]
        view_keys = [k for k in input_dict.keys() if re.match('extrinsic_rot_[0-4]',k)]
        # print (view_keys)
        view_keys.sort()
        
        assert (len(view_keys)==self.num_views)

        latent_3d_rot_all = []
        # cam_2_world = input_dict['extrinsic_rot_inv'].view( (batch_size, 3, 3) ).float()
        # world_2_cam = input_dict['extrinsic_rot'].    view( (batch_size, 3, 3) ).float()
        # cam2cam = torch.bmm(world_2_cam_shuffled, cam_2_world)
        cam_2_world = input_dict['extrinsic_rot_inv'].view( (batch_size, 3, 3) ).float()
        for k in view_keys:
            world_2_cam = input_dict[k].view( (batch_size, 3, 3) ).float()
            cam2cam = torch.bmm(world_2_cam, cam_2_world)
            latent_3d_rot_curr = torch.bmm(latent_3d, cam2cam.transpose(1,2))
            latent_3d_rot_all.append(latent_3d_rot_curr.view(batch_size,-1))

        latent_3d_rot_all = torch.cat(latent_3d_rot_all, dim = 1)
        # print ('latent_3d_rot_all',latent_3d_rot_all.size())
        return latent_3d_rot_all

    def forward_pain(self, input_dict):
        # print ('HIIIIIIIII')
        # print (input_dict.keys())
        # for k in input_dict.keys():
        #     print (k, input_dict[k].size())
        # s = raw_input()

        input = input_dict['img_crop']
        batch_size = input_dict['img_crop'].size()[0]
        device = input.device
        
        # latent_3d, latent_fg = self.base_network.get_encoder_outputs(input_dict)
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat).view(batch_size, -1, 3)
        latent_3d = self.get_latent_concatenated(latent_3d, input_dict)

        # print (self.to_pain)
        # s = raw_input()

        output_pain = self.to_pain.forward({'latent_3d': latent_3d})['3D']
        # output_pain = self.collate(output_pain, input_dict['segment_key'])
        pain_pred = torch.nn.functional.softmax(output_pain, dim = 1)

        ###############################################
        # Select the right output
        output_dict_all = {'pain': output_pain, 'pain_pred': pain_pred} 
        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict    

