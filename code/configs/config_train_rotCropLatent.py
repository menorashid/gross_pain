from configs.config_train_rotation_crop_newCal import *

config_dict['output_types'] = ['img_crop', 'latent_3d','latent_3d_transformed']
config_dict['loss_weight_latent'] = 1
config_dict['every_nth_frame'] = 1
config_dict['model_type'] = 'unet_encode3D_clean'