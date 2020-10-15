import os
import re
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import importlib
import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import torch.optim
#import pickle
import IPython

import train_encode_decode
from rhodin.python.utils import io as rhodin_utils_io
from rhodin.python.losses import generic as losses_generic
from metrics.binary_accuracy import BinaryAccuracy
from rhodin.python.utils import training as utils_train
from helpers import util

import math
import torch
import torch.optim

from rhodin.python.ignite._utils import convert_tensor
from rhodin.python.ignite.engine import Events

from multiview_dataset import MultiViewDataset
from seg_based_dataset import SegBasedSampler
from rhodin.python.utils import datasets as rhodin_utils_datasets

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
     

def get_loss(loss_type, config_dict, accuracy=False, deno_key = 'deno'):
    if loss_type=='cross_entropy':
        loss = losses_generic.LossLabel('pain', torch.nn.CrossEntropyLoss())

    if loss_type == 'cross_entropy':
            loss_train = losses_generic.LossLabel(pain_key, torch.nn.CrossEntropyLoss())
            loss_test = losses_generic.LossLabel(pain_key, torch.nn.CrossEntropyLoss())
    elif 'mil' in loss_type.lower():
        loss = getattr(losses_generic, loss_type)
        loss = loss('pain', 'segment_key', config_dict[deno_key], accuracy = accuracy)
        print (loss,loss.deno, loss.accuracy)
    else:
        raise ValueError('Loss type %s not allowed'%loss_type)

    
    return loss

class IgniteTrainPainFromLatent(train_encode_decode.IgniteTrainNVS):
    def __init__(self, config_dict_file, config_dict, config_dict_for_saved_model):
        
        self.config_dict_for_saved_model = config_dict_for_saved_model
        
        super().__init__(config_dict_file, config_dict)
        
        if self.model is None:
            return
        # redefine these
        self.trainer = utils_train.create_supervised_trainer(self.model, self.optimizer, self.loss_train, device=device, forward_fn = self.model.forward_pain, backward_every = config_dict.get('backward_every',1))
        self.evaluator = utils_train.create_supervised_evaluator(self.model,
                                                metrics=self.metrics,
                                                device=device, forward_fn = self.model.forward_pain)
        

    def load_network(self, config_dict):

        # load the base network without saved params
        assert 'pretrained_network_path' not in self.config_dict_for_saved_model.keys()
        network_base = super().load_network(self.config_dict_for_saved_model)

        # fill it with saved params
        pretrained_network_path = config_dict['pretrained_network_path']            
        pretrained_states = torch.load(pretrained_network_path, map_location=device)
        utils_train.transfer_partial_weights(pretrained_states, network_base, submodule=0) # last argument is to remove "network.single" prefix in saved network
        print("Done loading weights from config_dict['pretrained_network_path']", pretrained_network_path)
        # s = input()

        # define the pain model with pretrained encoder
        model_type_str = config_dict['model_type']    
        pain_model = importlib.import_module('models.'+model_type_str)
        network_pain = pain_model.PainHead(base_network = network_base, output_types = config_dict['output_types']) 
        print (network_pain.to_pain)
        # s = input()
        return network_pain

    
    def load_metrics(self, loss_test):
        loss_type = config_dict.get('loss_type', 'cross_entropy')
        metrics = {'AccumulatedLoss': utils_train.AccumulatedLoss(loss_test)}
        if loss_type == 'cross_entropy':
            metrics['accuracy'] = BinaryAccuracy()
        else:
            metrics['accuracy'] = utils_train.AccumulatedLoss(get_loss(loss_type, config_dict, accuracy = True, deno_key = 'deno_test'))    

        return metrics

    def initialize_wandb(self):
        wandb.init(config=config_dict, entity='egp', project='pain-classification')
        
    def loadOptimizer(self, network, config_dict):
        params_all_id = list(map(id, network.parameters()))
        params_painnet_id = list(map(id, network.to_pain.parameters()))
        params_toOptimize = [p for p in network.parameters() if id(p) in params_painnet_id]

        params_static_id = [id_p for id_p in params_all_id if not id_p in params_painnet_id]

        for p in params_toOptimize:
            print ('opt',id(p),p.size())
        # disable gradient computation for static params, saves memory and computation
        for p in network.parameters():
            if id(p) in params_static_id:
                p.requires_grad = False

        print("Normal learning rate: {} params".format(len(params_painnet_id)))
        print("Static learning rate: {} params".format(len(params_static_id)))
        print("Total: {} params".format(len(params_all_id)))

        opt_params = [{'params': params_toOptimize, 'lr': config_dict['learning_rate']}]
        optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        # s = input()
        return optimizer

    def load_loss(self, config_dict):
        # pain_key = 'pain'
        # segment_key = 'segment_key'
        loss_type = config_dict.get('loss_type', 'cross_entropy')
        # if loss_type == 'cross_entropy':
        loss_train = get_loss(loss_type, config_dict)
        loss_test = get_loss(loss_type, config_dict, deno_key = 'deno_test')
        #         def get_loss(loss_type, config_dict, accuracy=False, deno_key = 'deno')
        #     # losses_generic.LossLabel(pain_key, torch.nn.CrossEntropyLoss())
        #     loss_test = losses_generic.LossLabel(pain_key, torch.nn.CrossEntropyLoss())
        # elif loss_type == 'mil_loss':
        #     loss_train = losses_generic.MIL_Loss(pain_key, segment_key, config_dict['deno'])
        #     # , accuracy = True)
        #     loss_test = losses_generic.MIL_Loss(pain_key, segment_key, 8)
        # else:
        #     raise ValueError('Loss type %s not allowed'%loss_type)

        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self, config_dict, old = False):#, config_dict):
        return get_parameter_description_pain(config_dict, old)

    def load_data_train(self, config_dict, save_path):
        if 'seg' in config_dict['training_set'].lower():
            data_folder = config_dict['dataset_folder_train']
            input_types = config_dict['input_types']
            label_types = config_dict['label_types_train']
            subjects = config_dict['train_subjects']
            str_aft = config_dict['csv_str_aft']
            batch_size = config_dict['batch_size_train']
            rot_folder = config_dict.get('rot_folder',None)
            dataset = MultiViewDataset(data_folder=data_folder,
                                       bg_folder=data_folder,
                                       input_types=input_types,
                                       label_types=label_types,
                                       subjects=subjects,
                                       rot_folder = rot_folder,
                                       str_aft = str_aft)

            sampler = SegBasedSampler(data_folder, 
                         batch_size,
                         num_frames_per_seg = config_dict['num_frames_per_seg'],
                         subjects = subjects,
                         randomize=True,
                         every_nth_segment=config_dict['every_nth_frame'],
                         str_aft = str_aft,
                       min_size = config_dict['min_size_seg'])

            loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=config_dict['num_workers'], pin_memory=False,
                                                     collate_fn=rhodin_utils_datasets.default_collate_with_string)

        else:
            print ('hello train')
            loader =  super().load_data_train(config_dict, save_path)
            print ('super loader',loader)

        return loader

    def load_data_test(self, config_dict, save_path):
        if 'seg' in config_dict['training_set'].lower():
            data_folder = config_dict['dataset_folder_test']
            input_types = config_dict['input_types']
            label_types = config_dict['label_types_test']
            subjects = config_dict['test_subjects']
            str_aft = config_dict['csv_str_aft']
            batch_size = config_dict['batch_size_test']
            rot_folder = config_dict.get('rot_folder',None)
            dataset = MultiViewDataset(data_folder=data_folder,
                                       bg_folder=data_folder,
                                       input_types=input_types,
                                       label_types=label_types,
                                       subjects=subjects,
                                       rot_folder = rot_folder,
                                       str_aft = str_aft)

            sampler = SegBasedSampler(data_folder, 
                         batch_size,
                         num_frames_per_seg = config_dict['num_frames_per_seg'],
                         subjects = subjects,
                         randomize=False,
                         every_nth_segment=config_dict['every_nth_frame'],
                         str_aft = str_aft, 
                       min_size = config_dict['min_size_seg'])

            loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=config_dict['num_workers'], pin_memory=False,
                                                     collate_fn=rhodin_utils_datasets.default_collate_with_string)
        else:
            print ('hello test')
            loader =  super().load_data_test(config_dict, save_path)
            print ('super loader',loader)

        return loader
        

def get_parameter_description_pain(config_dict, old = False):
    shorter_train_subjects = [subject[:2] for subject in config_dict['train_subjects']]
    shorter_test_subjects = [subject[:2] for subject in config_dict['test_subjects']]

    if not old:
        shorter_train_subjects = '_'+util.join_string_list(shorter_train_subjects, '_')
        shorter_test_subjects  = '_'+util.join_string_list(shorter_test_subjects, '_')

    if 'new_folder_style' in config_dict.keys():
        folder = "../output/{model_type}_{loss_type}_{job_identifier}_{job_identifier_encdec}/{training_set}_nth_{every_nth_frame}_nfps_{num_frames_per_seg}/num_epochs_{num_epochs}_train{}_test{}_lr_{learning_rate}_backward_{backward_every}_bstrain_{batch_size_train}_bstest_{batch_size_test}".format(shorter_train_subjects, shorter_test_subjects,**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
    else:
        folder = "../output/trainNVSPainFromLatent_{job_identifier}_{job_identifier_encdec}/{training_set}/nth{every_nth_frame}_c{active_cameras}_train{}_test{}_lr{learning_rate}_bstrain{batch_size_train}_bstest{batch_size_test}".format(shorter_train_subjects, shorter_test_subjects,**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
    return folder


def get_model_path_pain(config_dict, epoch, old = False):
    folder = get_parameter_description_pain(config_dict, old)
    model_ext = 'network_0' + epoch + '.pth'
    model_path = os.path.join(folder, 'models', model_ext)
    # print ('Model Path',model_path)
    return model_path

def get_model_path(config_dict, epoch, old = False):
    folder = train_encode_decode.get_parameter_description(config_dict, old)
    model_ext = 'network_0' + epoch + '.pth'
    model_path = os.path.join(folder, 'models', model_ext)
    # print ('Model Path',model_path)
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
    # print(args)
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
    
    config_dict['pretrained_network_path'] = get_model_path(config_dict_for_saved_model, epoch=args.epoch_encdec)

    config_dict['rot_folder'] = config_dict_for_saved_model['rot_folder']
    config_dict['bg_folder'] =  config_dict_for_saved_model['bg_folder']
    ignite = IgniteTrainPainFromLatent(config_dict_module.__file__, config_dict, config_dict_for_saved_model)
    if ignite.model is not None:
        ignite.run()

