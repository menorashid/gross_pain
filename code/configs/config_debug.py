numJoints = 17
inputDimension = 128

config_dict = {
    # General parameters
    'dpi'                     : 190,
    'input_types'             : ['img_crop', 'bg_crop'],
    'output_types'            : ['img_crop'],
    'label_types_train'       : ['img_crop'],
    'label_types_test'        : ['img_crop'],
    'num_workers'             : 4,

    # opt parameters    
    'num_epochs'              : 1,
    'save_every'              : 100000,
    'learning_rate'           : 1e-3,# baseline: 0.001=1e-3
    'test_every'              : 5000,
    'plot_every'              : 5000,
    'print_every'             : 100,

    # LPS dataset parameters
    
    # views: [int] (which viewpoints to include in the dataset, e.g. all [0,1,2,3])
    #               The viewpoints are indexed starting from "front left" (FL=0) and
    #               then clock-wise -- FR=1, BR=2, BL=3. Front being the side facing
    #               the corridor, and R/L as defined from inside of the box.)

    'views'                   : '[0,1,2,3]',
    'image_width'             : 128,
    'image_height'            : 128,
    'data_dir_path'           : '../data/intervals_for_extraction_128_128_2fps/',

    # network parameters
    'batch_size_train' : 16,
    'batch_size_test' : 16, #10 #self.batch_size # Note, needs to be = self.batch_size for multi-view validation
    'outputDimension_3d' : numJoints * 3,
    'outputDimension_2d' : inputDimension // 8,

    # loss 
    'train_scale_normalized' : True,
    'train_crop_relative' : False,

    # dataset
    'bg_folder'            : '../data/median_bg/',
    'dataset_folder_train' : '../data/intervals_for_extraction_128_128_2fps/',
    'dataset_folder_test' : '../data/intervals_for_extraction_128_128_2fps/',
    'training_set' : 'LPS_2fps',
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
    'every_nth_frame' : 100,

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
    'skip_background' : True  # This means use background

}
