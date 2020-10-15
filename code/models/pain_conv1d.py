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
raw_input = input

# class TransposeForConv1d(nn.Module):
#     def forward(self, x):
#         x = x.unsqueeze(0).transpose(1,2)
#         return x

# class UnTransposeAfterConv1d(nn.Module):
#     def forward(self,x):
#         x = x.squeeze(0).transpose(0,1)
#         return x

class ConvSeg(nn.Module):
    def __init__(self, d_in, d_out, seq_len,relu,bnorm,dropout):
        super(ConvSeg, self).__init__()
        self.conv = torch.nn.Conv1d(d_in, d_out, seq_len, padding = seq_len//2)

        self.aft_conv = [nn.Identity()]
        if relu:
            self.aft_conv.append(torch.nn.ReLU())
        if bnorm:
            self.aft_conv.append(torch.nn.BatchNorm1d(d_out, affine=True))
        if dropout>0:
            self.aft_conv.append(nn.Dropout(dropout))

        # if len(self.aft_conv)>0:
        self.aft_conv = nn.Sequential(*self.aft_conv)

    def forward(self, x_all, segment_key):
        x_all_new = []
        for segment_val in torch.unique_consecutive(segment_key):
            segment_bin = segment_key==segment_val
            x = x_all[segment_bin,:]
            x = x.unsqueeze(0).transpose(1,2)
            x = self.conv(x)
            x = x.squeeze(0).transpose(0,1)
            x_all_new.append(x)
            # s = raw_input()
        
        x_all_new = torch.cat(x_all_new, axis = 0)
        x_all_new = self.aft_conv(x_all_new)

        return x_all_new
            # print (rel_latent.size())
            # h_n = self.lstm(rel_latent)
        

class PainHead(nn.Module):

    def __init__(self, base_network , output_types, n_hidden_to_pain = 2, d_hidden = 2048, dropout = 0.5 , seq_len = 9):
        super(PainHead, self).__init__()
        
        self.seq_len = seq_len

        self.encoder = base_network.encoder
        self.to_3d = base_network.to_3d

        conv_part = []
        d_in = base_network.dimension_3d
        d_out = d_hidden
        for i in range(n_hidden_to_pain):
            conv_part.append(ConvSeg(d_in, d_hidden, seq_len, relu=True, bnorm = True, dropout = dropout))
            d_in = d_hidden

        # conv_part = nn.Sequential(*conv_part)
        self.to_pain = nn.ModuleList(conv_part+[nn.Linear(d_hidden,2)])

        
        self.output_types = output_types

        
    def pad_input(self, latent_3d):
        seq_len = self.seq_len
        # batch_size = latent_3d.size(0)
        rem = seq_len - 1
        padding = torch.zeros((rem,latent_3d.size(1))).cuda()
        latent_3d = torch.cat((latent_3d,padding), axis = 0)
        return latent_3d
        

    def forward_pain(self, input_dict):
        # print ('HIIIIIIIII')
        
        input = input_dict['img_crop']
        segment_key = input_dict['segment_key']
        batch_size = input_dict['img_crop'].size()[0]
        # frame = input_dict['frame']

        device = input.device
        
        # latent_3d, latent_fg = self.base_network.get_encoder_outputs(input_dict)
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat)
        
        # for val in self.to_pain:
        #     print (type(val))

        for layer in self.to_pain[:-1]:
            latent_3d = layer(latent_3d, segment_key)

        output_pain = self.to_pain[-1](latent_3d)
        
        pain_pred = torch.nn.functional.softmax(output_pain, dim = 1)

        ###############################################
        # Select the right output
        output_dict_all = {'pain': output_pain, 'pain_pred': pain_pred, 'segment_key':segment_key} 
        output_dict = {}
        # print (self.output_types)
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict    

