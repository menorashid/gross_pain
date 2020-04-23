# Kinematic tree for humans, from h36m
# num_joints = 17
# bones = [[0, 1], [1, 2], [2, 3],
#              [0, 4], [4, 5], [5, 6],
#              [0, 7], [7, 8], [8, 9], [9, 10],
#              [8, 14], [14, 15], [15, 16],
#              [8, 11], [11, 12], [12, 13],
#             ]

# Kinematic tree for horses
num_joints = 36
joint_names = ['pelvis', 'pelvis0', 'spine', 'spine0', 'spine1', 'spine2', 'spine3', 'LLeg1', 'LLeg2', 'LLeg3', 'LFoot', 'RLeg1', 'RLeg2', 'RLeg3', 'RFoot', 'Neck', 'Neck1', 'Head', 'LLegBack1', 'LLegBack2', 'LLegBack3', 'LFootBack', 'RLegBack1', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7', 'Mouth', 'LEar', 'REar']
bones = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [6, 11], [11, 12], [12, 13], [13, 14], [6, 15], [15, 16], [16, 17], [0, 18], [18, 19], [19, 20], [20, 21], [0, 22], [22, 23], [23, 24], [24, 25], [0, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32], [17, 33], [17, 34], [17, 35]]

inputDimension = 128

config_dict = {
    # General parameters
    'dpi'                     : 190,
    # Possible input types    : 'img_crop' | 'bg_crop' | 'extrinsic_rot' | 'extrinsic_rot_inv'
    'input_types'             : ['img_crop', 'bg_crop','extrinsic_rot', 'extrinsic_rot_inv'],
    # Possible output types   :  'img_crop' | '3D'
    'output_types'            : ['img_crop'],
    # Possible lt train       : 'img_crop' | '3D' | 'bounding_box_cam' | 'intrinsic_crop' | 'extrinsic_rot' | 'extrinsic_rot_inv'
    'label_types_train'       : ['img_crop'],
    # Possible lt test        : 'img_crop' | '3D' | 'bounding_box_cam' | 'intrinsic_crop' | 'extrinsic_rot' | 'extrinsic_rot_inv'
    'label_types_test'        : ['img_crop'],
    'num_workers'             : 4,
    'bones'                   : bones,

    # opt parameters    
    'num_epochs'              : 50,
    'save_every'              : 5, #in epochs
    'learning_rate'           : 1e-3,# baseline: 0.001=1e-3
    'test_every'              : 1967, #size of epoch nth1 in iterations
    'plot_every'              : 1967,
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
    'bg_folder'            : '../data/median_bg/',
    'rot_folder': '../data/rotation_cal_1/',
    'training_set' : 'LPS_2fps_crop',
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
    'shuffle_fg' : True,
    'shuffle_3d' : True,
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

    # From rhodin config
    'note'              : 'resL3'

}
