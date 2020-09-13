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
from models import unet_encode3D_clean

class PainHead(nn.Module):

    def __init__(self, base_network , output_types, n_hidden_to_pain = 2, d_hidden = 2048, dropout = 0.5 ):
        super(PainHead, self).__init__()
        # self.base_network = base_network
        # attr_keep =['encoder','to_fg','to_3d','dimension_3d']
        # key_list = list(self.base_network.__dict__.keys())
        # for k in key_list:
        #     if k not in attr_keep:
        #         print (k)
        #         delattr(self.base_network, k)
        self.encoder = base_network.encoder
        self.to_3d = base_network.to_3d

        self.to_pain = MLP.MLP_fromLatent(d_in = base_network.dimension_3d, d_hidden=d_hidden, d_out=2, n_hidden=n_hidden_to_pain, dropout=dropout)
        # , no_bnorm = True)
        # print (self.to_pain)
        self.output_types = output_types
        # s = input()
        

    def forward_pain(self, input_dict):
        # print ('HIIIIIIIII')
        
        input = input_dict['img_crop']
        batch_size = input_dict['img_crop'].size()[0]
        device = input.device
        
        # latent_3d, latent_fg = self.base_network.get_encoder_outputs(input_dict)
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat).view(batch_size,-1,3)

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

