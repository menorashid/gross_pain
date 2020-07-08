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
raw_input = input

class unet(nn.Module):
    def __init__(self, feature_scale=4, # to reduce dimensionality
                 in_resolution=256,
                 output_channels=3, is_deconv=True,
                 upper_billinear=False,
                 lower_billinear=False,
                 in_channels=3, is_batchnorm=True, 
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
        super(unet, self).__init__()
        self.in_resolution = in_resolution
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.nb_stage = nb_stage
        self.dimension_bg=dimension_bg
        self.dimension_fg=dimension_fg
        self.dimension_3d=dimension_3d
        self.shuffle_fg=shuffle_fg
        self.shuffle_3d=shuffle_3d
        self.num_encoding_layers = num_encoding_layers
        self.output_types = output_types
        self.encoderType = encoderType
        assert dimension_3d % 3 == 0
        self.implicit_rotation = implicit_rotation
        self.num_cameras = num_cameras
        
        self.skip_connections = False
        self.skip_background = skip_background
        self.subbatch_size = subbatch_size
        self.latent_dropout = latent_dropout

        #filters = [64, 128, 256, 512, 1024]
        self.filters = [64, 128, 256, 512, 512, 512] # HACK
        self.filters = [int(x / self.feature_scale) for x in self.filters]
        self.bottleneck_resolution = in_resolution//(2**(num_encoding_layers-1))
        num_output_features = self.bottleneck_resolution**2 * self.filters[num_encoding_layers-1]
        print('bottleneck_resolution',self.bottleneck_resolution,'num_output_features',num_output_features)
        ns = 0
        ####################################
        ############ encoder ###############
        
        if self.encoderType == 'ResNet':
            self.encoder = resnet_VNECT_3Donly.resnet50(pretrained=True, input_key='img_crop', output_keys=['latent_3d','2D_heat'], 
                                                    input_width=in_resolution, num_classes=self.dimension_fg+self.dimension_3d)

        else:
            setattr(self, 'conv_1_stage' + str(ns), unetConv2(self.in_channels, self.filters[0], self.is_batchnorm, padding=1))
            setattr(self, 'pool_1_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            for li in range(2,num_encoding_layers): # note, first layer(li==1) is already created, last layer(li==num_encoding_layers) is created externally
                setattr(self, 'conv_'+str(li)+'_stage' + str(ns), unetConv2(self.filters[li-2], self.filters[li-1], self.is_batchnorm, padding=1))
                setattr(self, 'pool_'+str(li)+'_stage' + str(ns), nn.MaxPool2d(kernel_size=2))
            
            if from_latent_hidden_layers:
                setattr(self, 'conv_'+str(num_encoding_layers)+'_stage' + str(ns),  nn.Sequential( unetConv2(self.filters[num_encoding_layers-2], self.filters[num_encoding_layers-1], self.is_batchnorm, padding=1),
                                                                        nn.MaxPool2d(kernel_size=2)
                                                                        ))
            else:
                setattr(self, 'conv_'+str(num_encoding_layers)+'_stage' + str(ns), unetConv2(self.filters[num_encoding_layers-2], self.filters[num_encoding_layers-1], self.is_batchnorm, padding=1))

            module_list = []
            for li in range(1,self.num_encoding_layers): 
                module_list.append(getattr(self, 'conv_'+str(li)+'_stage' + str(ns)))
                module_list.append(getattr(self, 'pool_'+str(li)+'_stage' + str(ns)))
            module_list.append(getattr(self, 'conv_'+str(self.num_encoding_layers)+'_stage' + str(ns)))
            self.encoder = nn.Sequential(*module_list)
            
        ####################################
        ############ background ###############
        if skip_background:
            setattr(self, 'conv_1_stage_bg' + str(ns), unetConv2(self.in_channels, self.filters[0], self.is_batchnorm, padding=1))

        ###########################################################
        ############ latent transformation and pose ###############

        assert self.dimension_fg < self.filters[num_encoding_layers-1]
        num_output_features_3d = self.bottleneck_resolution**2 * (self.filters[num_encoding_layers-1] - self.dimension_fg)
        #setattr(self, 'fc_1_stage' + str(ns), Linear(num_output_features, 1024))
        setattr(self, 'fc_1_stage' + str(ns), Linear(self.dimension_3d, 128))
        setattr(self, 'fc_2_stage' + str(ns), Linear(128, num_joints * nb_dims))
        
        self.to_pose = MLP.MLP_fromLatent(d_in=self.dimension_3d, d_hidden=2048, d_out=150, n_hidden=n_hidden_to3Dpose, dropout=0.5)
        self.to_pain = MLP.MLP_fromLatent(d_in=self.dimension_3d, d_hidden=2048, d_out=2, n_hidden=n_hidden_to3Dpose, dropout=0.5)
                
        self.to_3d =  nn.Sequential( Linear(num_output_features, self.dimension_3d),
                                     Dropout(inplace=True, p=self.latent_dropout) # removing dropout degrades results
                                   )

        if self.implicit_rotation:
            print('WARNING: doing implicit rotation!')
            rotation_encoding_dimension = 128
            self.encode_angle =  nn.Sequential(Linear(3*3, rotation_encoding_dimension//2),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),
                                         Linear(rotation_encoding_dimension//2, rotation_encoding_dimension),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),
                                         Linear(rotation_encoding_dimension, rotation_encoding_dimension),
                                         )
            
            self.rotate_implicitely = nn.Sequential(Linear(self.dimension_3d + rotation_encoding_dimension, self.dimension_3d),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))
                   
        if from_latent_hidden_layers:
            hidden_layer_dimension = 1024
            if self.dimension_fg > 0:
                self.to_fg =  nn.Sequential( Linear(num_output_features, 256), # HACK pooling 
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),
                                         Linear(256, self.dimension_fg),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))
            self.from_latent =  nn.Sequential( Linear(self.dimension_3d, hidden_layer_dimension),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False),                        
                                         Linear(hidden_layer_dimension, num_output_features_3d),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))
        else:
            if self.dimension_fg > 0:
                self.to_fg =  nn.Sequential( Linear(num_output_features, self.dimension_fg),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))
            self.from_latent =  nn.Sequential( Linear(self.dimension_3d, num_output_features_3d),
                             Dropout(inplace=True, p=self.latent_dropout),
                             ReLU(inplace=False))


        ####################################
        ############ decoder ###############
        

        upper_conv = self.is_deconv and not upper_billinear
        lower_conv = self.is_deconv and not lower_billinear
        
        module_list = []
        for li in range(1,num_encoding_layers-1):
            setattr(self, 'upconv_'+str(li)+'_stage' + str(ns), unetUpNoSKip(self.filters[num_encoding_layers-li], self.filters[num_encoding_layers-li-1], upper_conv, padding=1))
            module_list.append(getattr(self, 'upconv_'+str(li)+'_stage' + str(ns)))
        self.decoder = nn.Sequential(*module_list)

        if self.skip_background:
            setattr(self, 'upconv_'+str(num_encoding_layers-1)+'_stage' + str(ns), unetUp(self.filters[1], self.filters[0], lower_conv, padding=1))
        else:
            setattr(self, 'upconv_'+str(num_encoding_layers-1)+'_stage' + str(ns), unetUpNoSKip(self.filters[1], self.filters[0], lower_conv, padding=1))
        
        setattr(self, 'final_stage' + str(ns), nn.Conv2d(self.filters[0], output_channels, 1))



    def get_shuff_idx(self, batch_size, rotation_by_user, device):

        ########################################################
        # Determine shuffling
        def shuffle_segment(list, start, end):
            selected = list[start:end]
            if self.training:
                if 0 and end-start == 2: # Note, was not enabled in ECCV submission, diabled now too HACK
                    prob = np.random.random([1])
                    if prob[0] > 1/self.num_cameras: # assuming four cameras, make it more often that one of the others is taken, rather than just autoencoding (no flip, which would happen 50% otherwise)
                        selected = selected[::-1] # reverse
                    else:
                        pass # let it as it is
                else:
                    random.shuffle(selected)

            else: # deterministic shuffling for testing
                selected = np.roll(selected,1).tolist()
            list[start:end] = selected

        def flip_segment(list, start, width):
            selected = list[start:start+width]
            list[start:start+width] = list[start+width:start+2*width]
            list[start+width:start+2*width] = selected
            
        shuffled_appearance = list(range(batch_size))
        shuffled_pose       = list(range(batch_size))
        num_pose_subbatches = batch_size//np.maximum(self.subbatch_size,1)
        
        
        if not rotation_by_user:
            if self.shuffle_fg and self.training==True:
                for i in range(0,num_pose_subbatches):
                    # Shuffle appearance samples randomly within subbatches
                    # For instance:
                    # [0, 1, 3, 2, 7, 4, 6, 5, 11, 8, 9, 10, 13, 12, 15, 14] 
                    shuffle_segment(shuffled_appearance, i*self.subbatch_size, (i+1)*self.subbatch_size)
                for i in range(0,num_pose_subbatches//2): # Switch first with second subbatch, for each pair of subbatches
                    # Result for above example:
                    # [7, 4, 6, 5, 0, 1, 3, 2, 13, 12, 15, 14, 11, 8, 9, 10] 
                    flip_segment(shuffled_appearance, i*2*self.subbatch_size, self.subbatch_size)
            if self.shuffle_3d:
                for i in range(0,num_pose_subbatches):
                    # Shuffle pose samples randomly within subbatches
                    # print('shuffled_pose      in if',shuffled_pose)
                    shuffle_segment(shuffled_pose, i*self.subbatch_size, (i+1)*self.subbatch_size)
                 
        # infer inverse mapping
        shuffled_pose_inv = [-1] * batch_size
        for i,v in enumerate(shuffled_pose):
            shuffled_pose_inv[v]=i
            
        # print('self.training',self.training,'shuffled_appearance',shuffled_appearance)
        # print('shuffled_pose      ',shuffled_pose)
        # s = raw_input()
            
        shuffled_appearance = torch.LongTensor(shuffled_appearance).to(device)
        shuffled_pose       = torch.LongTensor(shuffled_pose).to(device)
        shuffled_pose_inv   = torch.LongTensor(shuffled_pose_inv).to(device)

        if rotation_by_user:
            if 'shuffled_appearance' in input_dict.keys():
                shuffled_appearance = input_dict['shuffled_appearance'].long()
        
        return shuffled_appearance, shuffled_pose, shuffled_pose_inv 

    def get_cam2cam(self, input_dict, shuffled_pose, rotation_by_user):
        batch_size = input_dict['img_crop'].size()[0]
        cam_2_world = input_dict['extrinsic_rot_inv'].view( (batch_size, 3, 3) ).float()
        world_2_cam = input_dict['extrinsic_rot'].    view( (batch_size, 3, 3) ).float()
        if rotation_by_user:
            external_cam = input_dict['external_rotation_cam'].view(1,3,3).expand( (batch_size, 3, 3) )
            external_glob = input_dict['external_rotation_global'].view(1,3,3).expand( (batch_size, 3, 3) )
            cam2cam = torch.bmm(external_cam,torch.bmm(world_2_cam, torch.bmm(external_glob, cam_2_world)))
        else:
            world_2_cam_shuffled = torch.index_select(world_2_cam, dim=0, index=shuffled_pose)
            cam2cam = torch.bmm(world_2_cam_shuffled, cam_2_world)
            # print (cam2cam)
        return cam2cam


    def get_latent_3d_rotated(self, input_dict, latent_3d, cam2cam):
        if self.implicit_rotation:
            encoded_angle = self.encode_angle(cam2cam.view(batch_size,-1))
            encoded_latent_and_angle = torch.cat([latent_3d.view(batch_size,-1), encoded_angle], dim=1)
            latent_3d_rotated = self.rotate_implicitely(encoded_latent_and_angle)
        else:
            # print ('LADIES AND GENTLEMEN! WE ARE ROTATING!')
            latent_3d_rotated = torch.bmm(latent_3d, cam2cam.transpose(1,2))

        if 'shuffled_pose_weight' in input_dict.keys():
            w = input_dict['shuffled_pose_weight']
            # weighted average with the last one
            latent_3d_rotated = (1-w.expand_as(latent_3d))*latent_3d + w.expand_as(latent_3d)*latent_3d_rotated[-1:].expand_as(latent_3d)

        return latent_3d_rotated

    def get_encoder_outputs(self, input_dict):
        batch_size = input_dict['img_crop'].size()[0]
        latent_fg = None
        conv1_bg_shuffled = None
        ns=0
        has_fg = hasattr(self, 'to_fg')  # If latent_fg dim > 0
        if self.encoderType == 'ResNet':
            output = self.encoder.forward(input_dict)['latent_3d']
            if has_fg:
                latent_fg = output[:,:self.dimension_fg]
            latent_3d = output[:,self.dimension_fg:self.dimension_fg+self.dimension_3d].contiguous().view(batch_size,-1,3)
        else: # UNet encoder
            out_enc_conv = self.encoder(input_dict['img_crop'])
            center_flat = out_enc_conv.view(batch_size,-1)
            if has_fg:
                latent_fg = self.to_fg(center_flat)
            latent_3d = self.to_3d(center_flat).view(batch_size,-1,3)

        return latent_3d, latent_fg


    def get_decoder_output(self, batch_size, latent_3d_rotated, latent_fg_shuffled, conv1_bg_shuffled):
        ns =0
        map_from_3d = self.from_latent(latent_3d_rotated.view(batch_size,-1))
        map_width = self.bottleneck_resolution #out_enc_conv.size()[2]
        map_channels = self.filters[self.num_encoding_layers-1] #out_enc_conv.size()[1]
        if hasattr(self, 'to_fg'):
            latent_fg_shuffled_replicated = latent_fg_shuffled.view(batch_size,self.dimension_fg,1,1).expand(batch_size, self.dimension_fg, map_width, map_width)
            latent_shuffled = torch.cat([latent_fg_shuffled_replicated, map_from_3d.view(batch_size, map_channels-self.dimension_fg, map_width, map_width)], dim=1)
        else:
            latent_shuffled = map_from_3d.view(batch_size, map_channels, map_width, map_width)

        out_deconv = x = self.decoder(latent_shuffled)
        if self.skip_background:
            out_deconv = getattr(self, 'upconv_'+str(self.num_encoding_layers-1)+'_stage' + str(ns))(conv1_bg_shuffled, out_deconv)
        else:
            out_deconv = getattr(self, 'upconv_'+str(self.num_encoding_layers-1)+'_stage' + str(ns))(out_deconv)

        output_img_shuffled = getattr(self, 'final_stage' + str(ns))(out_deconv)

        return output_img_shuffled


    def forward(self, input_dict):
        if self.skip_connections:
            assert False

        input = input_dict['img_crop']
        device = input.device
        batch_size = input.size()[0]
        # num_pose_examples = batch_size//2
        # num_appearance_examples = batch_size//2
        # num_appearance_subbatches = num_appearance_examples//np.maximum(self.subbatch_size,1)
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
            latent_3d_transformed = torch.index_select(latent_3d_rotated, dim=0, index=shuffled_pose_inv)
        else:
            latent_3d_rotated = latent_3d
            latent_3d_transformed = latent_3d


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
        output_img = torch.index_select(output_img_shuffled, dim=0, index=shuffled_pose_inv)


        ###############################################
        # 3D pose stage (parallel to image decoder)
        output_pose = self.to_pose.forward({'latent_3d': latent_3d})['3D']
        output_pain = self.to_pain.forward({'latent_3d': latent_3d})['3D']


        ###############################################
        # Select the right output
        output_dict_all = {'3D' : output_pose, 'img_crop' : output_img, 'shuffled_pose' : shuffled_pose,
                           'shuffled_appearance' : shuffled_appearance, 'latent_3d': latent_3d, 'latent_3d_transformed': latent_3d_transformed,
                           'pain': output_pain } 
        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict
