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
import train_encode_decode_pain as tedp


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class IgniteTestPainFromLatent(tedp.IgniteTrainPainFromLatent):
    
    def get_out_file(self, config_dict, epochs):
        save_path = self.get_parameter_description(config_dict)
        # epochs = config_dict['num_epochs']
        out_file = os.path.join(save_path,'models','network_%03d.pth'%epochs)
        print (out_file)
        return out_file, save_path

    def __init__(self, config_dict_file, config_dict, config_dict_for_saved_model, epoch):
        
        self.config_dict_for_saved_model = config_dict_for_saved_model
        config_dict = self.set_up_config_dict(config_dict)
        wandb_run = self.set_up_wandb(config_dict)
        
        self.epoch_test = epoch

        out_file, self.save_path = self.get_out_file(config_dict, epoch)
        if not os.path.exists(out_file):
            print ('out_file',out_file,'does not exist. nothing to test')
            self.model = None
            return None

        self.out_file = out_file
        # _, self.loss_test = self.load_loss(config_dict)
        self.test_loader = self.load_data_test(config_dict, self.save_path)
        model = self.load_network(config_dict)
        
        self.model = model.to(device)
        self.metrics = self.load_metrics(config_dict)
        # self.metrics = {'accuracy': self.metrics['accuracy']}

        self.evaluator = utils_train.create_supervised_evaluator(self.model,
                                                metrics=self.metrics,
                                                device=device, forward_fn = self.model.forward_pain)

    def load_metrics(self,config_dict):
        metrics = {}
        loss_type = config_dict.get('loss_type')
        # metrics['thresh'] = utils_train.AccumulatedF1AndAccu(tedp.get_loss(loss_type, config_dict, accuracy = True, deno_key = 'deno_test'))    
        # metrics['argmax'] = utils_train.AccumulatedF1AndAccu(tedp.get_loss(loss_type, config_dict, accuracy = 'argmax', deno_key = 'deno_test'))    
        # metrics['old'] = utils_train.AccumulatedLoss(tedp.get_loss(loss_type, config_dict, accuracy = 'old', deno_key = 'deno_test'))    
        metrics['majority'] = utils_train.AccumulatedF1AndAccu(tedp.get_loss(loss_type, config_dict, accuracy = 'majority', deno_key = 'deno_test'))    
        
        return metrics


    def run(self):
        evaluator = self.evaluator
        save_path = self.save_path
        evaluator.run(self.test_loader, metrics=self.metrics)
        # print (evaluator.state.metrics['argmax_pain'])
        # return
        metrics = evaluator.state.metrics
        out_file = self.out_file.replace('.pth','_test.txt')
        lines = []
        for key in metrics.keys():
            str_to_print = [key]
            vals = metrics[key]
            if not hasattr(vals, '__iter__'):
                vals = [vals]
            for val in vals:
                str_to_print.append('%.4f'%val)
            str_to_print = ','.join(str_to_print)
            print (str_to_print)
            lines.append(str_to_print)
        util.writeFile(out_file, lines)


        

    def load_network(self, config_dict):
        
        out_file = self.out_file
        model_type_str = config_dict['model_type']    
        pain_model = importlib.import_module('models.'+model_type_str)
        
        network_base = self.load_base_network()

        if 'network_params' in config_dict.keys():
            network_pain = pain_model.PainHead(base_network = network_base, output_types = config_dict['output_types'],**config_dict['network_params']) 
        else:
            network_pain = pain_model.PainHead(base_network = network_base, output_types = config_dict['output_types']) 

        pretrained_states = torch.load(out_file, map_location=device)
        
        utils_train.transfer_partial_weights(pretrained_states, network_pain, submodule=0) # 

        # todo transfer weights here
        print (network_pain.to_pain)
        # print (pretrained_states)
        # s = input()
        return network_pain
 

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
    parser.add_argument('--epoch', type=int,
        help="Which epoch to test.")
    
    return parser.parse_args(argv)

def main(args):
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
    
    config_dict['pretrained_network_path'] = tedp.get_model_path(config_dict_for_saved_model, epoch=args.epoch_encdec)

    config_dict['rot_folder'] = config_dict_for_saved_model['rot_folder']
    config_dict['bg_folder'] =  config_dict_for_saved_model['bg_folder']
    ignite = IgniteTestPainFromLatent(config_dict_module.__file__, config_dict, config_dict_for_saved_model, epoch = args.epoch)
    if ignite.model is not None:
        ignite.run()
    
if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    print (args.epoch)
    main(args)
    # print(args)
    

