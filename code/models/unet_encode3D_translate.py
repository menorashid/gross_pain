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
import unet_encode3D_clean.unet as MasterUnet


class unet(MasterUnet):
    def __init__(self, feature_scale=4, # to reduce dimensionality
                 in_resolution=256,
                 output_channels=3, is_deconv=True,
                 upper_billinear=False,
                 lower_billinear=False,
                 in_channels=3, is_batchnorm=True, 
                 skip_background=True,
                 num_joints=17, nb_dims=3, # ecoding transformation
                 encoderType='UNet',
                 num_encoding_layers=5,
                 dimension_bg=256,
                 dimension_fg=256,
                 dimension_3d=3*64, # needs to be divisible by 3
                 latent_dropout=0.3,
                 shuffle_fg=True,
                 shuffle_3d=True,
                 from_latent_hidden_layers=0,
                 n_hidden_to3Dpose=2,
                 subbatch_size = 4,
                 implicit_rotation = False,
                 nb_stage=1, # number of U-net stacks
                 output_types=['3D', 'img_crop', 'shuffled_pose', 'shuffled_appearance' ],
                 num_cameras=4,
                 ):
    
        super(unet, self).__init__(feature_scale,
                                in_resolution,
                                output_channels,
                                upper_billinear,
                                lower_billinear,
                                in_channels,
                                skip_background,
                                num_joints,
                                encoderType,
                                num_encoding_layers,
                                dimension_bg,
                                dimension_fg,
                                dimension_3d,
                                latent_dropout,
                                shuffle_fg,
                                shuffle_3d,
                                from_latent_hidden_layers,
                                n_hidden_to3Dpose,
                                subbatch_size,
                                implicit_rotation,
                                nb_stage,
                                output_types,
                                num_cameras)

