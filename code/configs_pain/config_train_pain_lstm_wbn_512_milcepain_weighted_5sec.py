from rhodin.python.utils import io as rhodin_utils_io
import os

config_dict = {
    # model type
    'model_type' : 'pain_lstm_wbn_512',
    'new_folder_style' : True,

    # General parameters
    'dpi'                     : 190,
    'input_types'             : ['img_crop', 'segment_key'],
    'output_types'            : ['pain','segment_key'],
    'label_types_train'       : ['img_crop','pain','segment_key'],
    'label_types_test'        : ['img_crop','pain','segment_key'],
    'num_workers'             : 32,
    
    # opt parameters    
    'num_epochs'              : 10,
    'save_every'              : 5,
    'train_test_every'        : 5,
    'learning_rate'           : 1e-4,# baseline: 0.001=1e-3
    'test_every'              : 1,
    'plot_every'              : 100,
    'print_every'             : 10,
    'backward_every'          : 1,

    # LPS dataset parameters
    'views'                   : '[0,1,2,3]',
    'image_width'             : 128,
    'image_height'            : 128,

    # network parameters
    'batch_size_train' : 1200,
    'batch_size_test' : 1200,

    # loss 
    'loss_type' : 'MIL_Loss_Pain_CE',
    'loss_weighted': True,
    'accuracy_type' : ['argmax_pain'],
    'deno' : 'random',
    'deno_test' : 8,

    # dataset
    'training_set' : 'LPS_2fps_crop_timeseg',
    'csv_str_aft': '_reduced_2fps_frame_index_withSegIndexAndIntKey.csv',
    'num_frames_per_seg': 10, #10 min long segs
    'min_size_seg': 10,

    'img_mean' : (0.485, 0.456, 0.406),
    'img_std' : (0.229, 0.224, 0.225),
    'active_cameras' : False,
    'every_nth_frame' : 1,

    'project_wandb': 'debug',
}

