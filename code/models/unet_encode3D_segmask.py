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


class unet(unet_encode3D_clean.unet):
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
        assert not skip_background
        output_channels = 4
        super(unet, self).__init__(feature_scale=feature_scale, 
                                    in_resolution=in_resolution,
                                    output_channels=output_channels, is_deconv=is_deconv,
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
        self.sig = torch.nn.Sigmoid()

    def forward(self, input_dict):
        if self.skip_connections:
            assert False

        input = input_dict['img_crop']
        device = input.device
        batch_size = input.size()[0]
        rotation_by_user = self.training==False and 'external_rotation_cam' in input_dict.keys()
        
        ###############################################
        # determine shuffled rotation
        
        shuffled_appearance, shuffled_pose, shuffled_pose_inv = self.get_shuff_idx(batch_size, rotation_by_user, device)

        if 'extrinsic_rot' in input_dict.keys():
            cam2cam = self.get_cam2cam(input_dict, shuffled_pose, rotation_by_user)
        
            
        ###############################################
        # encoding stage
        
        latent_3d, latent_fg = self.get_encoder_outputs(input_dict)

        
        ###############################################
        # do shuffling
        
        if 'extrinsic_rot' in input_dict.keys():
            latent_3d_rotated = self.get_latent_3d_rotated(input_dict, latent_3d, cam2cam)
        else:
            latent_3d_rotated = latent_3d


        if self.skip_background:
            input_bg = input_dict['bg_crop'] # TODO take the rotated one/ new view
            input_bg_shuffled = torch.index_select(input_bg, dim=0, index=shuffled_pose)
            conv1_bg_shuffled = self.conv_1_stage_bg0(input_bg_shuffled)
        else:
            conv1_bg_shuffled = None

        if hasattr(self, 'to_fg'):
            latent_fg_shuffled = torch.index_select(latent_fg, dim=0, index=shuffled_appearance)        
            if 'shuffled_appearance_weight' in input_dict.keys():
                w = input_dict['shuffled_appearance_weight']
                latent_fg_shuffled = (1-w.expand_as(latent_fg))*latent_fg + w.expand_as(latent_fg)*latent_fg_shuffled
        else:
            latent_fg_shuffled = None

        
        ###############################################
        # decoding
        output_img_shuffled = self.get_decoder_output(batch_size, latent_3d_rotated, latent_fg_shuffled, conv1_bg_shuffled)


        ###############################################
        # de-shuffling
        output_decoder = torch.index_select(output_img_shuffled, dim=0, index=shuffled_pose_inv)

        #  this is the different part.
        mask = self.sig(output_decoder[:,3:4,:,:])
        output_img = mask*output_decoder[:,:3,:,:]+(1-mask)*input_dict['bg_crop']
        
        ###############################################
        # 3D pose stage (parallel to image decoder)
        output_pose = self.to_pose.forward({'latent_3d': latent_3d})['3D']
        output_pain = self.to_pain.forward({'latent_3d': latent_3d})['3D']


        ###############################################
        # Select the right output
        output_dict_all = {'3D' : output_pose, 'img_crop' : output_img, 'shuffled_pose' : shuffled_pose,
                           'shuffled_appearance' : shuffled_appearance, 'latent_3d': latent_3d,
                           'pain': output_pain, 'mask' : mask } 
        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict

    
