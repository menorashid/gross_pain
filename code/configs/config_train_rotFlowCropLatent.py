from configs.config_train_rotation_crop_newCal import *

config_dict['output_types'] = ['img_crop', 'latent_3d','latent_3d_transformed']
config_dict['loss_weight_latent'] = 0.1
config_dict['model_type'] = 'unet_encode3D_clean'
# config_dict['training_set'] = 'LPS_10fps_crop_oft_0.7'

config_dict['training_set'] = 'LPS_10fps_crop_ofp_0.01'
config_dict['csv_str_aft'] = '_reduced_percent_0.01_frame_index.csv'

config_dict['batch_size_train'] = 128
config_dict['batch_size_test'] = 128
config_dict['every_nth_frame'] = 1
config_dict['test_every'] = 1
config_dict['save_every'] = 10
config_dict['print_every'] = 10
