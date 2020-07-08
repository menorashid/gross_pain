# Kinematic tree for horses

num_joints = 50
joint_names = ['CristaFac_L', 'CristaFac_R', 'Poll', 'T8_', 'T12', 'T15','L3_','L5_','TubSac',
    'S5_', 'SpinaSca_R', 'Shoulder_R', 'Elbow_R', 'Carpus_R', 'MC3_R_PD', 'MC3_R_PP',
    'MC3_R_DD', 'MC3_R_DP', 'Fetl_RF', 'Hoof_RF', 'TubCox_R', 'Hip_R', 'Knee_R', 'Tarsus_R',
     'MT3_R_PD', 'MT3_R_PP', 'MT3_R_DD', 'MT3_R_DP', 'Fetl_RH', 'Hoof_RH', 'SpinaSca_L',
    'Shoulder_L', 'Elbow_L', 'Carpus_L', 'MC3_L_PD', 'MC3_L_PP', 'MC3_L_DD', 'MC3_L_DP',
    'Fetl_LF', 'Hoof_LF', 'TubCox_L', 'Hip_L', 'Knee_L', 'Tarsus_L', 'MT3_L_PD',
     'MT3_L_PP', 'MT3_L_DD', 'MT3_L_DP', 'Fetl_LH', 'Hoof_LH', 'Sternum']
bones=[[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [2, 9], [9, 10], [10, 11],
 [11, 12], [12, 13], [13, 15], [15, 17], [12, 14], [14, 16], [16, 17], [17, 18], [2, 29], [29, 30],
 [30, 31], [31, 32], [32, 33], [33, 35], [35, 37], [32, 34], [34, 36], [36, 37], [37, 38],
 [8, 20], [20, 21], [21, 22], [22, 23], [23, 25], [25, 27], [22, 24], [24, 26], [26, 27], [27, 28],
 [8, 40], [40, 41], [41, 42], [42, 43], [43, 45], [45, 47], [42, 44], [44, 46], [46, 47],
 [47, 48], [8, 49]]


inputDimension = 128

config_dict = {
    # General parameters
    'dpi'                     : 190,
    'input_types'             : ['img_crop', 'bg_crop'],
    # Possible output types   :  'img_crop' | '3D' | 'shuffled_pose' | 'shuffled_appearance' | 'latent_3d'
    'output_types'            : ['3D'],
    'label_types_train'       : ['3D', 'pose_mean', 'pose_std'],
    'label_types_test'        : ['3D', 'pose_mean', 'pose_std'],
    'num_workers'             : 4,
    'bones'                   : bones,

# Classfication
    'num_joints' : 50,

    # opt parameters    
    'num_epochs'              : 1,
    'save_every'              : 1,
    'learning_rate'           : 1e-3,# baseline: 0.001=1e-3
    'train_test_every'        : 1,
    'test_every'              : 1,
    'plot_every'              : 2510,
    'print_every'             : 10,

    # LPS dataset parameters
    
    # views: [int] (which viewpoints to include in the dataset, e.g. all [0,1,2,3])
    #               The viewpoints are indexed starting from "front left" (FL=0) and
    #               then clock-wise -- FR=1, BR=2, BL=3. Front being the side facing
    #               the corridor, and R/L as defined from inside of the box.)

    'views'                   : [0,1,2,3],
    # 'image_width'             : 720,
    # 'image_height'            : 576,
    'image_width'             : 128,
    'image_height'            : 128,

    # network parameters
    'batch_size_train' : 64,
    'batch_size_test' : 64, #10 #self.batch_size # Note, needs to be = self.batch_size for multi-view validation
#     'outputDimension_3d' : num_joints * 3,
    'outputDimension_2d' : inputDimension // 8,

    # loss 
    'train_scale_normalized' : True,
    'train_crop_relative' : False,

    # dataset
    'training_set' : 'treadmill',
    'project_wandb': '3d-pose-estimation',
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
    'every_nth_frame' : 500,

    # Encoder-decoder
    'model_type': 'unet_encode3D_clean',
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
    'skip_background' : True  # This means use background

}
