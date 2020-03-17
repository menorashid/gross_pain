config_dict = {
    'input_types'             : ['img'],
    'output_types'            : ['3D', 'img'],
    'label_types_train'       : ['img_crop'],
    'label_types_test'        : ['img_crop'],
    'num_workers'             : 4,

    # opt parameters    
    'num_training_iterations' : 600000,
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
    'data_dir_path'           : '../data/frames_debug/',
    'train_subjects'          : ['aslan', 'inkasso'],

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
    'skip_background' : True
}
