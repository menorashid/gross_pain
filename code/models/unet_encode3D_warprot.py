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
from . import unet_encode3D_clean
raw_input = input

class unet(unet_encode3D_clean.unet):

    def get_cam2cam(self, input_dict, shuffled_pose, rotation_by_user):
        batch_size = input_dict['img_crop'].size()[0]
        cam_2_world = input_dict['extrinsic_rot_inv'].view( (batch_size, 3, 3) ).float()
        world_2_cam = input_dict['extrinsic_rot'].    view( (batch_size, 3, 3) ).float()
        warp_rot = input_dict['warp_rot'].view( (batch_size, 3, 3) ).float()
        warp_rot_inv = warp_rot.transpose(1,2)
        
        # print (warp_rot[0],warp_rot[5])
        # print (warp_rot_inv[0],warp_rot_inv[5])

        cam_2_world = torch.bmm(cam_2_world,warp_rot_inv)
        world_2_cam = torch.bmm(warp_rot, world_2_cam)
        # raw_input()
        assert not rotation_by_user
        if rotation_by_user:
            external_cam = input_dict['external_rotation_cam'].view(1,3,3).expand( (batch_size, 3, 3) )
            external_glob = input_dict['external_rotation_global'].view(1,3,3).expand( (batch_size, 3, 3) )
            cam2cam = torch.bmm(external_cam,torch.bmm(world_2_cam, torch.bmm(external_glob, cam_2_world)))
        else:
            world_2_cam_shuffled = torch.index_select(world_2_cam, dim=0, index=shuffled_pose)

            cam2cam = torch.bmm(world_2_cam_shuffled, cam_2_world)
            # print (cam2cam)

        # print (cam2cam.size())
        # print (cam2cam[:10])
        # print (cam_2_world)
        # s = input()
        return cam2cam
