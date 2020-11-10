from configs.config_train_rotation_crop_newCal import *

config_dict['model_type'] = 'unet_encode3D_warprot'
config_dict['input_types'] += ['warp_rot']
config_dict['training_set'] = 'LPS_10fps_crop_ofp_0.01'
config_dict['csv_str_aft'] = '_reduced_percent_0.01_frame_index.csv'
config_dict['bg_post_pend'] = '_bg_month'

config_dict['batch_size_train'] = 128
config_dict['batch_size_test'] = 128
config_dict['every_nth_frame'] = 1
config_dict['test_every'] = 1
config_dict['save_every'] = 10
config_dict['print_every'] = 10
config_dict['plot_test_every'] = 1

# 'save_every'              : 50, #in epochs
#     'test_every'              : 1, #size of epoch nth1 in iterations

# config_dict['learning_rate']= 1e-2# baseline: 0.001=1e-3