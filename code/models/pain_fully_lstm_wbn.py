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

class PainHead(nn.Module):

    def __init__(self, base_network , output_types, n_hidden_to_pain = 1, d_hidden = 1024, dropout = 0.5 , seq_len = 10, all_out = False):
        super(PainHead, self).__init__()
        
        self.seq_len = seq_len
        self.all_out = all_out
        self.encoder = base_network.encoder
        self.to_3d = base_network.to_3d
        if n_hidden_to_pain>1:
            self.lstm = nn.LSTM(input_size = base_network.dimension_3d , hidden_size = d_hidden, num_layers = n_hidden_to_pain, dropout = dropout)
        else:
            self.lstm = nn.LSTM(input_size = base_network.dimension_3d , hidden_size = d_hidden, num_layers = n_hidden_to_pain)

        self.to_pain = nn.ModuleList([self.lstm,nn.Sequential(torch.nn.BatchNorm1d(d_hidden, affine=True),nn.Dropout()),nn.LSTM(input_size = d_hidden , hidden_size = 2, num_layers = 1)])

        # print (self.to_pain)
        self.output_types = output_types
        # s = raw_input()

        
    def pad_input(self, latent_3d):
        seq_len = self.seq_len
        input_size = latent_3d.size(0)
        rem = 1 if (input_size%seq_len)>0 else 0
        batch_size = input_size//seq_len + rem

        rem = seq_len*batch_size - input_size
        padding = torch.zeros((rem,latent_3d.size(1))).cuda()
        latent_3d = torch.cat((latent_3d,padding), axis = 0)
        
        latent_3d = torch.reshape(latent_3d,(batch_size,seq_len,latent_3d.size(1)))
        latent_3d = torch.transpose(latent_3d, 0,1)
        return latent_3d
        

    def forward_pain(self, input_dict):
        # print ('HIIIIIIIII')
        
        input = input_dict['img_crop']
        segment_key = input_dict['segment_key']
        batch_size = input_dict['img_crop'].size()[0]
        device = input.device
        
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat)
        
        h_n_all = []
        segment_key_new = []
        for segment_val in torch.unique_consecutive(segment_key):
            segment_bin = segment_key==segment_val
            rel_latent = latent_3d[segment_bin,:]
            init_size = rel_latent.size(0)
            rel_latent = self.pad_input(rel_latent)
            out_lstm,(h_n,c_n) = self.lstm(rel_latent)
            bef_size = out_lstm.size()
            out_lstm = out_lstm.view(out_lstm.size(0)*out_lstm.size(1), out_lstm.size(2))
            out_lstm = self.to_pain[1](out_lstm)
            out_lstm = out_lstm.view(bef_size)
            out_pain,(h_n,c_n) = self.to_pain[2](out_lstm)
            if self.all_out:
                # print (out_lstm.size(), init_size)
                out_pain = out_pain.view(out_pain.size(0)*out_pain.size(1),-1)
                out_pain = out_pain[:init_size,:]
                h_n = out_pain[:init_size,:]
                
            else:
                h_n = h_n[-1]

            # print (h_n.size())
            h_n_all.append(h_n)
            

            segment_key_rel = segment_key[segment_bin][:h_n.size(0)]
            segment_key_new.append(segment_key_rel)

        output_pain = torch.cat(h_n_all,axis = 0)
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

