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
from models import unet_encode3D_segmask, unet_encode3D_translate

class unet(unet_encode3D_segmask.unet, unet_encode3D_translate.unet):
    def __init__(self, 
                feature_scale=4, 
                 in_resolution=256,
                 is_deconv=True,
                 upper_billinear=False,
                 lower_billinear=False,
                 in_channels=3, 
                 is_batchnorm=True, 
                 skip_background=True,
                 num_joints=17, 
                 nb_dims=3, # ecoding transformation
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

        super(unet, self).__init__(feature_scale=feature_scale, 
                                    in_resolution=in_resolution,
                                    is_deconv=is_deconv,
                                    upper_billinear=upper_billinear,
                                    lower_billinear=lower_billinear,
                                    in_channels=in_channels, is_batchnorm=is_batchnorm, 
                                    skip_background=skip_background,
                                    num_joints=num_joints, 
                                    nb_dims=nb_dims, 
                                    encoderType=encoderType,
                                    num_encoding_layers=num_encoding_layers,
                                    dimension_bg=dimension_bg,
                                    dimension_fg=dimension_fg,
                                    dimension_3d=dimension_3d, # needs to be divisible by 3
                                    latent_dropout=latent_dropout,
                                    shuffle_fg=shuffle_fg,
                                    shuffle_3d=shuffle_3d,
                                    from_latent_hidden_layers=from_latent_hidden_layers,
                                    n_hidden_to3Dpose=n_hidden_to3Dpose,
                                    subbatch_size =subbatch_size ,
                                    implicit_rotation =implicit_rotation ,
                                    nb_stage=nb_stage, 
                                    output_types=output_types,
                                    num_cameras=num_cameras)

    
