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
raw_input = input

class PainHead(PainSimple):

    def forward_pain(self, input_dict):
        # print ('HIIIIIIIII')
        
        input = input_dict['img_crop']
        batch_size = input_dict['img_crop'].size()[0]
        device = input.device
        
        # latent_3d, latent_fg = self.base_network.get_encoder_outputs(input_dict)
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat).view(batch_size, -1, 3)
        cam2world = input_dict['extrinsic_rot_inv'].view( (batch_size, 3, 3) ).float()
        latent_3d = torch.bmm(latent_3d, cam2world.transpose(1,2))
        latent_3d = latent_3d.view(batch_size,-1,3)
        
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

