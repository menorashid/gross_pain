import os
import re
from helpers import util, visualize
import numpy as np
import pandas as pd
import numpy as np
import imageio
import torch
import sklearn.manifold
import sklearn.preprocessing

import train_encdec_pose as tep
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import io as rhodin_utils_io
from rhodin.python.utils import plot_dict_batch as utils_plot_batch

from train_encode_decode_pain import get_model_path 
import glob

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)


class IgniteTestNVS(tep.IgniteTrainPoseFromLatent):
    def run(self, config_dict):
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
        data_loader = self.load_data_test(config_dict)
        model = self.load_network(config_dict)
        model = model.to(device)

        return model, data_loader, config_dict

def set_up_config_dict(config_file,
                     train_subjects,
                     test_subjects,
                     job_identifier,
                     batch_size_test,
                     dataset_path,
                     input_to_get, 
                     output_to_get,
                     train_subjects_model,
                     test_subjects_model,
                     config_file_model,
                     job_identifier_encdec,
                     epoch_encdec
                     ):
    config_dict_for_saved_model = rhodin_utils_io.loadModule(config_file_model).config_dict
    config_dict_for_saved_model['implicit_rotation'] = config_dict_for_saved_model.get('implicit_rotation', False)
    config_dict_for_saved_model['skip_background'] = config_dict_for_saved_model.get('skip_background', True)
    config_dict_for_saved_model['loss_weight_pose3D'] = config_dict_for_saved_model.get('loss_weight_pose3D', 0)
    config_dict_for_saved_model['n_hidden_to3Dpose'] = config_dict_for_saved_model.get('n_hidden_to3Dpose', 2)
    config_dict_for_saved_model['train_subjects'] = train_subjects_model
    config_dict_for_saved_model['test_subjects'] = test_subjects_model
    config_dict_module = rhodin_utils_io.loadModule(config_file)
    config_dict = config_dict_module.config_dict
    config_dict_for_saved_model['job_identifier'] = job_identifier_encdec
    config_dict['job_identifier_encdec'] = job_identifier_encdec
    config_dict['job_identifier'] = job_identifier
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    config_dict['data_dir_path'] = dataset_path
    config_dict['dataset_folder_train'] = dataset_path
    config_dict['dataset_folder_test'] = dataset_path
    config_dict['dataset_folder_mocap'] = os.path.join(dataset_path, 'treadmill_lameness_mocap_ci_may11/mocap/')
    config_dict['dataset_folder_rgb'] = os.path.join(dataset_path, 'animals_data/')
    root = dataset_path.rsplit('/', 2)[0]
    config_dict['bg_folder'] = config_dict_for_saved_model['bg_folder']
    config_dict['rot_folder'] = config_dict_for_saved_model['rot_folder']
    config_dict['pretrained_network_path'] = tep.get_model_path(config_dict_for_saved_model, epoch=epoch_encdec)
    assert os.path.exists(config_dict['pretrained_network_path'])
    for val in input_to_get:
        if val not in config_dict['input_types']:
            config_dict['input_types'].append(val) 

    config_dict['output_types'] = []
    for val in output_to_get:
        if val not in config_dict['output_types']:
            config_dict['output_types'].append(val) 
    return config_dict

def swap_bg_treadmill():
    config_path = 'configs/config_pose_debug.py'
    train_subjects = ['HOR', 'LAC', 'LAR', 'LAZ', 'LEA', 'LOR', 'PRA']
    test_subjects = ['ART']
    
    job_identifier = 'testing_bg_swap'
    batch_size_test = 64
    dataset_path = '../data'
    config_path = 'configs/config_pose_debug.py'
    job_identifier = 'withRotCropNewCal'

    output_to_get = ['img_crop']
    input_to_get = ['img_crop']

    train_subjects_model = 'brava/herrera/inkasso/julia/kastanjett/naughty_but_nice/sir_holger'.split('/')
    test_subjects_model = ['aslan']
    epoch_encdec = str(50)
    config_file_model = 'configs/config_train_rotation_newCal.py'
    job_identifier_encdec = 'withRotNewCal'

    config_dict = set_up_config_dict(config_path, train_subjects, test_subjects, job_identifier, batch_size_test, dataset_path, input_to_get,  output_to_get, train_subjects_model, test_subjects_model, config_file_model, job_identifier_encdec, epoch_encdec)

    ignite = IgniteTestNVS()
    model, data_loader, config_dict = ignite.run(config_dict)
    data_iterator = iter(data_loader)

    for input_dict, label_dict in data_iterator:
        input_dict['bg_crop'] = torch.ones(input_dict['img_crop'].size())
        # torch.max(input_dict['img_crop'])
        input_dict_cuda, label_dict_cuda = rhodin_utils_datasets.nestedDictToDevice((input_dict, label_dict), device=device)
        output_dict_cuda = model(input_dict_cuda)
        utils_plot_batch.plot_iol(input_dict_cuda, label_dict_cuda, output_dict_cuda, config_dict, 'train', '../scratch/try_it_out_samebg.jpg')
        break    

def main():
    swap_bg_treadmill()
    return
    import latent_space_nn as lsn
    dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
    config_path = 'configs/config_train_rotation_newCal.py'
    job_identifier = 'withRotNewCal'
    out_dir = '../scratch/rot'
    nth_frame = 10

    # dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
    # config_path = 'configs/config_train_rotation_crop_newCal.py'
    # job_identifier = 'withRotCropNewCal'
    # out_dir = '../scratch/rotcrop'
    # nth_frame = 1

    util.mkdir(out_dir)

    model_num = 50
    batch_size_test = 64
    output_to_get = ['img_crop']
    input_to_get = ['img_crop']
    train_subjects = 'brava/herrera/inkasso/julia/kastanjett/naughty_but_nice/sir_holger'.split('/')
    test_subject = 'aslan'
    all_subjects = train_subjects+[test_subject]

    config_dict = lsn.set_up_config_dict(config_path, train_subjects, [test_subject], job_identifier, batch_size_test, dataset_path, input_to_get, output_to_get)
    model_path = lsn.get_model_path(config_dict, str(model_num))
    
    config_dict['pretrained_network_path'] = model_path
    config_dict['every_nth_frame'] = nth_frame

    ignite = lsn.IgniteTestNVS()
    model, data_loader, config_dict = ignite.run(config_path, config_dict)
    data_iterator = iter(data_loader)

    for input_dict, label_dict in data_iterator:
        # input_dict['bg_crop'] = torch.max(input_dict['img_crop'])*torch.ones(input_dict['img_crop'].size())
        input_dict_cuda, label_dict_cuda = rhodin_utils_datasets.nestedDictToDevice((input_dict, label_dict), device=device)
        output_dict_cuda = model(input_dict_cuda)
        utils_plot_batch.plot_iol(input_dict_cuda, label_dict_cuda, output_dict_cuda, config_dict, 'train', os.path.join(out_dir,'lps_true.jpg'))

        input_dict['bg_crop'] = 0*torch.ones(input_dict['img_crop'].size())
        input_dict_cuda, label_dict_cuda = rhodin_utils_datasets.nestedDictToDevice((input_dict, label_dict), device=device)
        output_dict_cuda = model(input_dict_cuda)
        utils_plot_batch.plot_iol(input_dict_cuda, label_dict_cuda, output_dict_cuda, config_dict, 'train', os.path.join(out_dir,'lps_white.jpg'))
        break    


if __name__=='__main__':
    main()