import os
import re
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

import sys, os, shutil

from rhodin.python.utils import io as rhodin_utils_io
import numpy as np
import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import torch.optim
#import pickle
import IPython

import train_encode_decode
from rhodin.python.losses import generic as losses_generic
from rhodin.python.losses import poses as losses_poses
from metrics.binary_accuracy import BinaryAccuracy
from rhodin.python.utils import training as utils_train

class IgniteTrainPainFromLatent(train_encode_decode.IgniteTrainNVS):
    def load_metrics(self, loss_test):
        metrics = {'loss': utils_train.AccumulatedLoss(loss_test),
                   'accuracy': BinaryAccuracy()}
        return metrics

    def initialize_wandb(self):
        wandb.init(config=config_dict, entity='egp', project='pain-classification')
        
    def loadOptimizer(self, network, config_dict):
        params_all_id = list(map(id, network.parameters()))
        params_painnet_id = list(map(id, network.to_pain.parameters()))
        params_toOptimize = [p for p in network.parameters() if id(p) in params_painnet_id]

        params_static_id = [id_p for id_p in params_all_id if not id_p in params_painnet_id]

        # disable gradient computation for static params, saves memory and computation
        for p in network.parameters():
            if id(p) in params_static_id:
                p.requires_grad = False

        print("Normal learning rate: {} params".format(len(params_painnet_id)))
        print("Static learning rate: {} params".format(len(params_static_id)))
        print("Total: {} params".format(len(params_all_id)))

        opt_params = [{'params': params_toOptimize, 'lr': config_dict['learning_rate']}]
        optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        return optimizer

    def load_loss(self, config_dict):
    
        pain_key = 'pain'
        loss_train = losses_generic.LossLabel(pain_key, torch.nn.CrossEntropyLoss())
        loss_test = losses_generic.LossLabel(pain_key, torch.nn.CrossEntropyLoss())

        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self, config_dict):#, config_dict):
        shorter_train_subjects = [subject[:2] for subject in config_dict['train_subjects']]
        shorter_test_subjects = [subject[:2] for subject in config_dict['test_subjects']]
        folder = "../output/trainNVSPainFromLatent_{job_identifier}_{job_identifier_encdec}/{training_set}/nth{every_nth_frame}_c{active_cameras}_train{}_test{}_lr{learning_rate}_bstrain{batch_size_train}_bstest{batch_size_test}".format(shorter_train_subjects, shorter_test_subjects,**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
        return folder


def get_model_path(config_dict, epoch):
    folder = train_encode_decode.get_parameter_description(config_dict)
    model_ext = 'network_0' + epoch + '.pth'
    model_path = os.path.join(folder, 'models', model_ext)
    print ('Model Path',model_path)
    return model_path


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
        help="Python file with config dictionary.")
    parser.add_argument('--config_file_model', type=str,
        help="Python file with config dictionary.")
    parser.add_argument('--dataset_path', type=str,
        help="Path to root folder for dataset.")
    parser.add_argument('--train_subjects', type=str,
        help="Which subjects to train on.")
    parser.add_argument('--test_subjects', type=str,
        help="Which subjects to test on.")
    parser.add_argument('--train_subjects_model', type=str,
        help="Which subjects to train on.")
    parser.add_argument('--test_subjects_model', type=str,
        help="Which subjects to test on.")
    parser.add_argument('--job_identifier', type=str,
        help="Slurm job ID, or other identifier, to not overwrite output.")
    parser.add_argument('--job_identifier_encdec', type=str,
        help="Job identifier for the saved model to load.")
    parser.add_argument('--epoch_encdec', type=str,
        help="Which epoch for the saved model to load.")
    return parser.parse_args(argv)
        
    
if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print(args)
    train_subjects = re.split('/', args.train_subjects)
    test_subjects = re.split('/',args.test_subjects)
    train_subjects_model = re.split('/', args.train_subjects_model)
    test_subjects_model = re.split('/',args.test_subjects_model)

    config_dict_for_saved_model = rhodin_utils_io.loadModule(args.config_file_model).config_dict
    config_dict_for_saved_model['implicit_rotation'] = config_dict_for_saved_model.get('implicit_rotation', False)
    config_dict_for_saved_model['skip_background'] = config_dict_for_saved_model.get('skip_background', True)
    config_dict_for_saved_model['loss_weight_pose3D'] = config_dict_for_saved_model.get('loss_weight_pose3D', 0)
    config_dict_for_saved_model['n_hidden_to3Dpose'] = config_dict_for_saved_model.get('n_hidden_to3Dpose', 2)

    config_dict_for_saved_model['train_subjects'] = train_subjects_model
    config_dict_for_saved_model['test_subjects'] = test_subjects_model

    config_dict_module = rhodin_utils_io.loadModule(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict_for_saved_model['job_identifier'] = args.job_identifier_encdec
    config_dict['job_identifier_encdec'] = args.job_identifier_encdec
    config_dict['job_identifier'] = args.job_identifier
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    config_dict['data_dir_path'] = args.dataset_path
    config_dict['dataset_folder_train'] = args.dataset_path
    config_dict['dataset_folder_test'] = args.dataset_path
    root = args.dataset_path.rsplit('/', 2)[0]
    config_dict['bg_folder'] = os.path.join(root, 'median_bg/')
    config_dict['rot_folder'] = os.path.join(root, 'rotation_cal_1/')

    config_dict['pretrained_network_path'] = get_model_path(config_dict_for_saved_model, epoch=args.epoch_encdec)

    ignite = IgniteTrainPainFromLatent()
    ignite.run(config_dict_module.__file__, config_dict)

