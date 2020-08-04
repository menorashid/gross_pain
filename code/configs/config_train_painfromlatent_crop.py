from rhodin.python.utils import io as rhodin_utils_io
import os

num_joints = 17
bones = [[0, 1], [1, 2], [2, 3],
             [0, 4], [4, 5], [5, 6],
             [0, 7], [7, 8], [8, 9], [9, 10],
             [8, 14], [14, 15], [15, 16],
             [8, 11], [11, 12], [12, 13],
            ]

inputDimension = 128
config_dict = {
    # General parameters
    'dpi'                     : 190,
    'input_types'             : ['img_crop', 'bg_crop'],
    # Possible output types   :  'img_crop' | '3D' | 'shuffled_pose' | 'shuffled_appearance' | 'latent_3d'
    'output_types'            : ['pain',],
    'label_types_train'       : ['img_crop','pain'],
    'label_types_test'        : ['img_crop','pain'],
    'num_workers'             : 4,
    'bones'                   : bones,

    # opt parameters    
    'num_epochs'              : 10,
    'save_every'              : 1,
    'train_test_every'        : 10,
    'learning_rate'           : 1e-3,# baseline: 0.001=1e-3
    'test_every'              : 1,
    'plot_every'              : 1,
    'print_every'             : 10,


    # LPS dataset parameters
    
    # views: [int] (which viewpoints to include in the dataset, e.g. all [0,1,2,3])
    #               The viewpoints are indexed starting from "front left" (FL=0) and
    #               then clock-wise -- FR=1, BR=2, BL=3. Front being the side facing
    #               the corridor, and R/L as defined from inside of the box.)

    'views'                   : '[0,1,2,3]',
    'image_width'             : 128,
    'image_height'            : 128,

    # network parameters
    'batch_size_train' : 64,
    'batch_size_test' : 64, #10 #self.batch_size # Note, needs to be = self.batch_size for multi-view validation
    'outputDimension_3d' : num_joints * 3,
    'outputDimension_2d' : inputDimension // 8,

    # loss 
    'train_scale_normalized' : True,
    'train_crop_relative' : False,

    # dataset
    'training_set' : 'LPS_2fps_crop',
    # 'training_set' : 'LPS_2fps',
    'img_mean' : (0.485, 0.456, 0.406),
    'img_std' : (0.229, 0.224, 0.225),
    'active_cameras' : False,
    'inputDimension' : inputDimension,
    'mirror_augmentation' : False,
    'perspectiveCorrection' : True,
    'rotation_augmentation' : True,
    'shear_augmentation' : 0,
    'scale_augmentation' : False,
    'seam_scaling' : 1.0,
    'use_view_batches' : 4,
    'use_subject_batches' : True,
    'every_nth_frame' : 1,

    # Encoder-decoder
    'latent_bg' : 0,
    'latent_fg' : 24,
    'latent_3d' : 200*3,
    'latent_dropout' : 0.3,
    'from_latent_hidden_layers' : 0,
    'upsampling_bilinear' : 'upper',
    'shuffle_fg' : False,
    'shuffle_3d' : False,
    'feature_scale' : 4,
    'num_encoding_layers' : 4,
    'loss_weight_rgb' : 1,
    'loss_weight_gradient' : 0.01,
    'loss_weight_imageNet' : 2,
    'loss_weight_3d' : 0,
    'do_maxpooling' : False,
    'encoderType' : 'UNet',
    'implicit_rotation' : False,
    'predict_rotation' : False,
    'skip_background' : True,  # This means use background

    'project_wandb': 'debug',
}

