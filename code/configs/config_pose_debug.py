inputDimension = 128

config_dict = {
    # General parameters
    'dpi'                     : 190,
    'input_types'             : ['img_crop'],
    # Possible output types   :  'img_crop' | '3D' | 'shuffled_pose' | 'shuffled_appearance' | 'latent_3d'
    'output_types'            : ['pose'],
    'label_types_train'       : ['pose'],
    'label_types_test'        : ['pose'],
    'num_workers'             : 4,
#     'bones'                   : bones,

# Classfication
    'num_joints' : 50,

    # opt parameters    
    'num_epochs'              : 1,
    'save_every'              : 100000,
    'learning_rate'           : 1e-3,# baseline: 0.001=1e-3
    'test_every'              : 2510,
    'plot_every'              : 2510,
    'print_every'             : 100,

    # LPS dataset parameters
    
    # views: [int] (which viewpoints to include in the dataset, e.g. all [0,1,2,3])
    #               The viewpoints are indexed starting from "front left" (FL=0) and
    #               then clock-wise -- FR=1, BR=2, BL=3. Front being the side facing
    #               the corridor, and R/L as defined from inside of the box.)

    'views'                   : [0,1,2,3],
    'image_width'             : 128,
    'image_height'            : 128,

    # network parameters
    'batch_size_train' : 48,
    'batch_size_test' : 48, #10 #self.batch_size # Note, needs to be = self.batch_size for multi-view validation
#     'outputDimension_3d' : num_joints * 3,
    'outputDimension_2d' : inputDimension // 8,

    # loss 
    'train_scale_normalized' : True,
    'train_crop_relative' : False,

    # dataset
    'training_set' : 'treadmill',
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

}
