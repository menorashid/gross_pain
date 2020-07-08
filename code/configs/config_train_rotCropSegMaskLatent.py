from configs.config_train_rotCrop_segmask import *

config_dict['output_types'] += ['latent_3d','latent_3d_transformed']
config_dict['loss_weight_latent'] = 1