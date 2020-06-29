import os
import re
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

import sys, os, shutil

from models import unet_encode3D_512
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

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

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

    def load_network(self, config_dict):
        output_types= config_dict['output_types']
        
        use_billinear_upsampling = config_dict.get('upsampling_bilinear', False)
        lower_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'half'
        upper_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'upper'
        
        from_latent_hidden_layers = config_dict.get('from_latent_hidden_layers', 0)
        num_encoding_layers = config_dict.get('num_encoding_layers', 4)
        
        num_cameras = 4
        if config_dict['active_cameras']: # for H36M it is set to False
            num_cameras = len(config_dict['active_cameras'])
        
        if lower_billinear:
            use_billinear_upsampling = False
        network_single = unet_encode3D_512.unet(dimension_bg=config_dict['latent_bg'],
                                            dimension_fg=config_dict['latent_fg'],
                                            dimension_3d=config_dict['latent_3d'],
                                            feature_scale=config_dict['feature_scale'],
                                            shuffle_fg=config_dict['shuffle_fg'],
                                            shuffle_3d=config_dict['shuffle_3d'],
                                            latent_dropout=config_dict['latent_dropout'],
                                            in_resolution=config_dict['inputDimension'],
                                            encoderType=config_dict['encoderType'],
                                            is_deconv=not use_billinear_upsampling,
                                            upper_billinear=upper_billinear,
                                            lower_billinear=lower_billinear,
                                            from_latent_hidden_layers=from_latent_hidden_layers,
                                            n_hidden_to3Dpose=config_dict['n_hidden_to3Dpose'],
                                            num_encoding_layers=num_encoding_layers,
                                            output_types=output_types,
                                            subbatch_size=config_dict['use_view_batches'],
                                            implicit_rotation=config_dict['implicit_rotation'],
                                            skip_background=config_dict['skip_background'],
                                            num_cameras=num_cameras,
                                            )

        if 'pretrained_network_path' in config_dict.keys(): # automatic
            if config_dict['pretrained_network_path'] == 'MPII2Dpose':
                pretrained_network_path = '/cvlabdata1/home/rhodin/code/humanposeannotation/output_save/CVPR18_H36M/TransferLearning2DNetwork/h36m_23d_crop_relative_s1_s5_aug_from2D_2017-08-22_15-52_3d_resnet/models/network_000000.pth'
                print("Loading weights from MPII2Dpose")
                pretrained_states = torch.load(pretrained_network_path, map_location=device)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0, add_prefix='encoder.') # last argument is to remove "network.single" prefix in saved network
            else:
                print("Loading weights from config_dict['pretrained_network_path']")
                pretrained_network_path = config_dict['pretrained_network_path']            
                pretrained_states = torch.load(pretrained_network_path, map_location=device)
                utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0) # last argument is to remove "network.single" prefix in saved network
                print("Done loading weights from config_dict['pretrained_network_path']")
        
        if 'pretrained_posenet_network_path' in config_dict.keys(): # automatic
            print("Loading weights from config_dict['pretrained_posenet_network_path']")
            pretrained_network_path = config_dict['pretrained_posenet_network_path']            
            pretrained_states = torch.load(pretrained_network_path, map_location=device)
            utils_train.transfer_partial_weights(pretrained_states, network_single.to_pose, submodule=0) # last argument is to remove "network.single" prefix in saved network
            print("Done loading weights from config_dict['pretrained_posenet_network_path']")

        # print (network_single)
        # s = input()
        return network_single
    
    

def get_model_path(config_dict, epoch):
    shorter_train_subjects = [subject[:2] for subject in config_dict['train_subjects']]
    shorter_test_subjects = [subject[:2] for subject in config_dict['test_subjects']]

    folder = "../output/trainNVS_{job_identifier}_{encoderType}_layers{num_encoding_layers}_implR{implicit_rotation}_w3Dp{loss_weight_pose3D}_w3D{loss_weight_3d}_wRGB{loss_weight_rgb}_wGrad{loss_weight_gradient}_wImgNet{loss_weight_imageNet}/skipBG{skip_background}_bg{latent_bg}_fg{latent_fg}_3d{latent_3d}_lh3Dp{n_hidden_to3Dpose}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_{training_set}/nth{every_nth_frame}_c{active_cameras}_train{}_test{}_bs{use_view_batches}_lr{learning_rate}".format(shorter_train_subjects, shorter_test_subjects,**config_dict)
    folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
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

