from rhodin.python.utils import io as rhodin_utils_io
import os
config_dict = rhodin_utils_io.loadModule("configs/config_debug.py").config_dict

config_dict['label_types_test']  += ['pain']
config_dict['label_types_train'] += ['pain']
config_dict['latent_dropout'] = 0

config_dict['shuffle_fg'] = False
config_dict['shuffle_3d'] = False
config_dict['actor_subset'] = [1]
config_dict['useCamBatches'] = 0
config_dict['useSubjectBatches'] = 0
config_dict['train_scale_normalized'] = 'mean_std'

# pose training on full dataset
#config_dict['actor_subset'] = [1,5,6,7,8]

