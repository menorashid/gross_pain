from configs.config_train_rotation_crop_newCal import *

config_dict['output_types'] = ['img_crop', 'latent_3d','latent_3d_transformed']
config_dict['loss_weight_latent'] = 1
config_dict['every_nth_frame'] = 1
config_dict['model_type'] = 'unet_encode3D_clean'
config_dict['model_type'] = 'unet_encode3D_clean'
config_dict['training_set'] = 'LPS_10fps_crop_oft_0.7'
config_dict['batch_size_train'] = 128
config_dict['batch_size_test'] = 128
