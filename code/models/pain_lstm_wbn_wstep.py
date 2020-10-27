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

class PainHead(pain_lstm_wbn.PainHead):

    # def __init__(self):
    #     super(PainHead, self).__init__()

    def __init__(self, base_network , output_types, n_hidden_to_pain = 1, d_hidden = 1024, dropout = 0.5 , seq_len = 10, step_size = 5):
        super(PainHead, self).__init__(base_network , output_types, n_hidden_to_pain = n_hidden_to_pain, d_hidden = d_hidden, dropout = dropout , seq_len = seq_len)
        self.step_size = step_size
        print (self.step_size,self.seq_len)
        
    def pad_input(self, latent_3d):
        seq_len = self.seq_len
        step_size = self.step_size

        # print (latent_3d.size())
        input_size = latent_3d.size(0)
        rem = 1 if (input_size%seq_len)>0 else 0
        batch_size = input_size//seq_len + rem

        rem = seq_len*batch_size - input_size
        padding = torch.zeros((rem,latent_3d.size(1))).cuda()
        latent_3d = torch.cat((latent_3d,padding), axis = 0)
        
        latent_3d = [latent_3d[start:start+seq_len] for start in range(0,len(latent_3d)-seq_len+1,step_size)]
        assert (len(latent_3d[-1])==seq_len)
        batch_size = len(latent_3d)
        latent_3d = torch.cat(latent_3d,axis = 0)

        latent_3d = torch.reshape(latent_3d,(batch_size,seq_len,latent_3d.size(1)))
        latent_3d = torch.transpose(latent_3d, 0,1)
        # print (batch_size)
        # print (latent_3d.size())
        return latent_3d
        
