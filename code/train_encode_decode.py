import os
import re
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiview_dataset import MultiViewDataset, MultiViewDatasetCrop, MultiViewDatasetSampler
import os, shutil

import numpy as np
import importlib
# from models import unet_encode3D_clean as unet_encode3D
from rhodin.python.losses import generic as losses_generic
from rhodin.python.losses import images as losses_images
from rhodin.python.ignite.metrics import Loss
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import io as rhodin_utils_io
from rhodin.python.utils import training as utils_train
from helpers import util

import math
import torch
# torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import torch.optim

# sys.path.insert(0,'./ignite')
from rhodin.python.ignite._utils import convert_tensor
from rhodin.python.ignite.engine import Events

import wandb

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class IgniteTrainNVS:
    
    def set_up_config_dict(self, config_dict):
        config_dict['implicit_rotation'] = config_dict.get('implicit_rotation', False)
        config_dict['skip_background'] = config_dict.get('skip_background', True)
        config_dict['loss_weight_pose3D'] = config_dict.get('loss_weight_pose3D', 0)
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
        config_dict['team_wandb'] = config_dict.get('team_wandb', 'egp')
        return config_dict
    
    def set_up_wandb(self, config_dict):
        if not config_dict['project_wandb']=='debug':
            wandb_run = self.initialize_wandb(config_dict)
        else:
            wandb_run = False
        return wandb_run

    def __init__(self, config_dict_file, config_dict,resume = 0):
        # print ('resume',resume)
        # return
        config_dict = self.set_up_config_dict(config_dict)
        wandb_run = self.set_up_wandb(config_dict)

    
        
        # save path and config files
        save_path = self.get_parameter_description(config_dict)
        epochs = config_dict['num_epochs']
        out_file = os.path.join(save_path,'models','network_%03d.pth'%epochs)
        if os.path.exists(out_file):
            print ('out_file',out_file,'exists. change exp params')
            self.model = None
            return None
        else:
            print ('we are good. out_file',out_file,'does not exist')
            # return None

        self.resume = resume
        if not self.resume:
            rhodin_utils_io.savePythonFile(config_dict_file, save_path)
            rhodin_utils_io.savePythonFile(__file__, save_path)
        
        model = self.load_network(config_dict)
        optimizer = self.loadOptimizer(model,config_dict)
       
        
        model = model.to(device)
        
        # now do training stuff
        
        self.train_loader = self.load_data_train(config_dict, save_path)
        test_loader = self.load_data_test(config_dict, save_path)
        
        
        loss_train,loss_test = self.load_loss(config_dict)
        metrics = self.load_metrics(loss_test, config_dict)
                    
        trainer = utils_train.create_supervised_trainer(model, optimizer, loss_train, device=device)
        evaluator = utils_train.create_supervised_evaluator(model,
                                                metrics=metrics,
                                                device=device)

        self.config_dict = config_dict
        self.save_path = save_path
        self.epochs = epochs
        self.wandb_run = wandb_run
        # self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_train = loss_train
        self.loss_test = loss_test
        self.metrics = metrics
        self.trainer = trainer
        self.evaluator = evaluator

    def run(self):
        config_dict = self.config_dict
        save_path = self.save_path
        epochs = self.epochs
        wandb_run = self.wandb_run
        train_loader = self.train_loader
        test_loader = self.test_loader
        model = self.model
        optimizer = self.optimizer
        loss_train = self.loss_train
        loss_test = self.loss_test
        metrics = self.metrics
        trainer = self.trainer
        evaluator = self.evaluator
        
        @trainer.on(Events.STARTED)
        def load_stuff(engine):
            if self.resume:
                strings_to_load = ['network_%03d.pth','optimizer_%03d.pth','state_%03d.pickle']
                strings_to_load = [os.path.join('models',string_curr%self.resume) for string_curr in strings_to_load]
                for val in strings_to_load:
                    print (val)
                    assert os.path.exists(os.path.join(save_path,val))
                print (engine.state)
                utils_train.load_model_state(save_path, model, optimizer, engine.state, strings_to_load)
                engine.state.epoch = self.resume
            # trainer.state.epoch = 

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_progress(engine):
            # log the loss
            iteration = engine.state.iteration - 1
            if iteration % config_dict['print_every'] == 0:
                utils_train.save_training_error(save_path, engine)
            # if iteration in [0,100]:
            #     utils_train.save_training_example(save_path, engine, config_dict)

        @trainer.on(Events.EPOCH_COMPLETED)
        def plot_training_image(engine):
            # log batch example image
            print ('plotting')
            epoch = engine.state.epoch - 1
            if epoch % config_dict['plot_every'] == 0:
                utils_train.save_training_example(save_path, engine, config_dict)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            ep = trainer.state.epoch
            if  'train_test_every' in config_dict.keys() and (ep) % config_dict['train_test_every'] == 0:
                print("Running evaluation of whole train set at epoch ", ep)
                evaluator.run(train_loader, metrics=metrics)
                _ = utils_train.save_testing_error(save_path, trainer, evaluator,
                                    dataset_str='Training set',
                                    save_extension='debug_log_whole_trainset.txt')

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate_model(engine):
            ep = engine.state.epoch
            if ((ep) % config_dict['test_every'] == 0) or ep==1: # +1 to prevent evaluation at iteration 0
                    # return
                print("Running evaluation at epoch ", ep)
                evaluator.run(test_loader, metrics=metrics)
                accumulated_loss = utils_train.save_testing_error(save_path, engine, evaluator,
                                    dataset_str='Validation set',
                                    save_extension='debug_log_testing.txt')
            
            if 'val_every' in config_dict.keys() and (ep) % config_dict['val_every'] == 0:
                print("Running validation at epoch ", ep)
                evaluator.run(self.val_loader, metrics=metrics)
                accumulated_loss = utils_train.save_testing_error(save_path, engine, evaluator,
                                    dataset_str='Validation set',
                                    save_extension='debug_log_validation.txt')
                if 'metric_for_save' in config_dict.keys():
                    metric_for_save = config_dict['metric_for_save']
                    val = evaluator.state.metrics[metric_for_save]
                    # print ('val')

                    if type(val)==tuple:
                        val = val[-1]
                    # val = 1-val
                    # evaluator.state.metrics['best_val'] = val

                    # save the best model
                    utils_train.save_model_state(save_path, trainer, val, model, optimizer, engine.state, wandb_run, max_it = True)
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def save_model(engine):
            epoch = engine.state.epoch
            print ('epoch',epoch,'engine.state.iteration',engine.state.iteration)
            if not epoch % config_dict['save_every']: # +1 to prevent evaluation at iteration 0
                utils_train.save_model_state_iter(save_path, trainer, model,
                                                  optimizer, engine.state, wandb_run)
            if not epoch % config_dict['num_epochs']: # +1 to prevent evaluation at iteration 0
                utils_train.save_model_state_iter(save_path, trainer, model,
                                                  optimizer, engine.state, wandb_run)

        # print test result
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_test_example(engine):
            print ('plotting test')
            epoch = engine.state.epoch - 1
            if ('plot_test_every' in config_dict.keys()) and (epoch % config_dict['plot_test_every'] == 0):
                utils_train.save_test_example(save_path, trainer, evaluator, config_dict)
        
        # kick everything off
        trainer.run(train_loader, max_epochs=epochs, metrics=metrics)

    def load_metrics(self, loss, config_dict= None):
        return {'AccumulatedLoss': utils_train.AccumulatedLoss(loss)}

    def initialize_wandb(self, config_dict):
        wandb_run = wandb.init(config=config_dict, job_type='train',
                               entity=config_dict['team_wandb'],
                               project=config_dict['project_wandb'])
        return wandb_run
        
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

        if 'model_type' not in config_dict:
            model_type_str = 'unet_encode3D_clean'
        else:
            model_type_str = config_dict['model_type']
        unet_encode3D = importlib.import_module('models.'+model_type_str)
        network_single = unet_encode3D.unet(dimension_bg=config_dict['latent_bg'],
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
        return network_single
    
    def loadOptimizer(self,network, config_dict):
        if network.encoderType == "ResNet":
            params_all_id = list(map(id, network.parameters()))
            params_resnet_id = list(map(id, network.encoder.parameters()))
            params_except_resnet = [i for i in params_all_id if i not in params_resnet_id]
            
            # for the more complex setup
            params_toOptimize_id = (params_except_resnet
                             + list(map(id, network.encoder.layer4_reg.parameters()))
                             + list(map(id, network.encoder.layer3.parameters()))
                             + list(map(id, network.encoder.l4_reg_toVec.parameters()))
                             + list(map(id, network.encoder.fc.parameters())))
            params_toOptimize    = [p for p in network.parameters() if id(p) in params_toOptimize_id]
    
            params_static_id = [id_p for id_p in params_all_id if not id_p in params_toOptimize_id]
    
            # disable gradient computation for static params, saves memory and computation
            for p in network.parameters():
                if id(p) in params_static_id:
                    p.requires_grad = False
    
            print("Normal learning rate: {} params".format(len(params_toOptimize_id)))
            print("Static learning rate: {} params".format(len(params_static_id)))
            print("Total: {} params".format(len(params_all_id)))
    
            opt_params = [{'params': params_toOptimize, 'lr': config_dict['learning_rate']}]
            optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=config_dict['learning_rate'])
        return optimizer
    
    def load_data_train(self,config_dict,save_path):
        str_aft = config_dict.get('csv_str_aft', None)
        bg_post_pend = config_dict.get('bg_post_pend', None)
        if 'crop' in config_dict['training_set'].lower():
            dataset = MultiViewDatasetCrop(data_folder=config_dict['dataset_folder_train'],
                                       bg_folder=config_dict['bg_folder'],
                                       input_types=config_dict['input_types'],
                                       label_types=config_dict['label_types_train'],
                                       subjects=config_dict['train_subjects'],
                                       rot_folder = config_dict['rot_folder'],
                                       str_aft = str_aft,
                                       bg_post_pend = bg_post_pend)
        else:
            dataset = MultiViewDataset(data_folder=config_dict['dataset_folder_train'],
                                       bg_folder=config_dict['bg_folder'],
                                       input_types=config_dict['input_types'],
                                       label_types=config_dict['label_types_train'],
                                       subjects=config_dict['train_subjects'],
                                       rot_folder = config_dict['rot_folder'],
                                       str_aft = str_aft)

        batch_sampler = MultiViewDatasetSampler(data_folder=config_dict['dataset_folder_train'],
                                                save_path=save_path,
                                                mode='train',
                                                subjects=config_dict['train_subjects'],
                                                use_subject_batches=config_dict['use_subject_batches'],
                                                use_view_batches=config_dict['use_view_batches'],
                                                batch_size=config_dict['batch_size_train'],
                                                randomize=True,
                                                every_nth_frame=config_dict['every_nth_frame'],
                                                str_aft = str_aft)

        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=config_dict['num_workers'], pin_memory=False,
                                             collate_fn=rhodin_utils_datasets.default_collate_with_string)
        return loader
    
    def load_data_test(self,config_dict,save_path):
        str_aft = config_dict.get('csv_str_aft', None)
        bg_post_pend = config_dict.get('bg_post_pend', None)
        if 'crop' in config_dict['training_set'].lower():
            dataset = MultiViewDatasetCrop(data_folder=config_dict['dataset_folder_test'],
                                       bg_folder=config_dict['bg_folder'],
                                       input_types=config_dict['input_types'],
                                       label_types=config_dict['label_types_test'],
                                       subjects=config_dict['test_subjects'],
                                       rot_folder = config_dict['rot_folder'],
                                       str_aft = str_aft,
                                       bg_post_pend = bg_post_pend)
        else:
            dataset = MultiViewDataset(data_folder=config_dict['dataset_folder_test'],
                                       bg_folder=config_dict['bg_folder'],
                                       input_types=config_dict['input_types'],
                                       label_types=config_dict['label_types_test'],
                                       subjects=config_dict['test_subjects'],
                                       rot_folder = config_dict['rot_folder'],
                                       str_aft = str_aft)

        batch_sampler = MultiViewDatasetSampler(data_folder=config_dict['dataset_folder_test'],
                                                save_path=save_path,
                                                mode='test',
                                                subjects=config_dict['test_subjects'],
                                                use_subject_batches=0,
                                                use_view_batches=config_dict['use_view_batches'],
                                                batch_size=config_dict['batch_size_test'],
                                                randomize=False,
                                                every_nth_frame=config_dict['every_nth_frame'],
                                                str_aft = str_aft)

        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=config_dict['num_workers'], pin_memory=False,
                                             collate_fn=rhodin_utils_datasets.default_collate_with_string)
        return loader
    
    def load_loss(self, config_dict):
        # normal
        if config_dict.get('MAE', False):
            pairwise_loss = torch.nn.modules.loss.L1Loss()
        else:
            pairwise_loss = torch.nn.modules.loss.MSELoss()
        image_pixel_loss = losses_generic.LossOnDict(key='img_crop', loss=pairwise_loss)
        
        image_imgNet_bare = losses_images.ImageNetCriterium(criterion=pairwise_loss, weight=config_dict['loss_weight_imageNet'], do_maxpooling=config_dict.get('do_maxpooling',True))
        image_imgNet_loss = losses_generic.LossOnDict(key='img_crop', loss=image_imgNet_bare)
        
        latent_dist_loss = losses_generic.LossOnPredDict(key = 'latent_3d', 
            other_key = 'latent_3d_transformed',
            loss = torch.nn.modules.loss.MSELoss(),
            weight = config_dict['loss_weight_latent'] if 'loss_weight_latent' in config_dict.keys() else 1.)
        
        if 'loss_opt_flow' in config_dict.keys() and config_dict['loss_opt_flow']>0:
            loss_opt_flow = losses_generic.LossWeightedOnDict(key = 'img_crop', weight_key = 'opt_flow',loss_weight = config_dict['loss_opt_flow'])

        losses_train = []
        losses_test = []
        
        if 'img_crop' in config_dict['output_types']:
            if config_dict['loss_weight_rgb']>0:
                losses_train.append(image_pixel_loss)
                losses_test.append(image_pixel_loss)
            if config_dict['loss_weight_imageNet']>0:
                losses_train.append(image_imgNet_loss)
                losses_test.append(image_imgNet_loss)
            if 'loss_weight_latent' in config_dict.keys() and config_dict['loss_weight_latent']>0:
                losses_train.append(latent_dist_loss)
                losses_test.append(latent_dist_loss)
            if 'loss_opt_flow' in config_dict.keys() and config_dict['loss_opt_flow']>0:
                losses_train.append(loss_opt_flow)
                losses_test.append(loss_opt_flow)
                

                
        loss_train = losses_generic.PreApplyCriterionListDict(losses_train, sum_losses=True)
        loss_test  = losses_generic.PreApplyCriterionListDict(losses_test,  sum_losses=True)
                
        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self, config_dict, old = False):
        return get_parameter_description(config_dict,old)
    
def get_parameter_description(config_dict, old = False):
    nb_chars = 3 if config_dict['training_set'] == 'treadmill' else 2
    shorter_train_subjects = [subject[:nb_chars] for subject in config_dict['train_subjects']]
    shorter_test_subjects = [subject[:nb_chars] for subject in config_dict['test_subjects']]
    
    train_subj_str = util.join_string_list(shorter_train_subjects, '_')
    test_subj_str = util.join_string_list(shorter_test_subjects, '_')
    
    if 'model_type' not in config_dict: #backward compatibility
        folder = "../output/trainNVS_{job_identifier}_{encoderType}_layers{num_encoding_layers}_implR{implicit_rotation}_w3Dp{loss_weight_pose3D}_w3D{loss_weight_3d}_wRGB{loss_weight_rgb}_wGrad{loss_weight_gradient}_wImgNet{loss_weight_imageNet}/skipBG{skip_background}_bg{latent_bg}_fg{latent_fg}_3d{latent_3d}_lh3Dp{n_hidden_to3Dpose}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_{training_set}/nth{every_nth_frame}_c{active_cameras}_train_{}_test_{}_bs{use_view_batches}_lr{learning_rate}".format(train_subj_str, test_subj_str, **config_dict)
    else: #added model type here
        folder = "../output/train_{model_type}_{job_identifier}_{encoderType}_layers{num_encoding_layers}_implR{implicit_rotation}_w3Dp{loss_weight_pose3D}_w3D{loss_weight_3d}_wRGB{loss_weight_rgb}_wGrad{loss_weight_gradient}_wImgNet{loss_weight_imageNet}/skipBG{skip_background}_bg{latent_bg}_fg{latent_fg}_3d{latent_3d}_lh3Dp{n_hidden_to3Dpose}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_{training_set}/nth{every_nth_frame}_c{active_cameras}_train_{}_test_{}_bs{use_view_batches}_lr{learning_rate}".format(train_subj_str, test_subj_str, **config_dict)

    if old:
        split_folder = os.path.split(folder)
        end_str = "nth{every_nth_frame}_c{active_cameras}_train{}_test{}_bs{use_view_batches}_lr{learning_rate}".format(shorter_train_subjects, shorter_test_subjects, **config_dict)        
        folder = os.path.join(split_folder[0],end_str)



    folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
    # print (folder)
    return folder


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
        help="Python file with config dictionary.")
    parser.add_argument('--dataset_path', type=str,
        help="Path to root folder for dataset.")
    parser.add_argument('--train_subjects', type=str,
        help="Which subjects to train on.")
    parser.add_argument('--test_subjects', type=str,
        help="Which subjects to test on.")
    parser.add_argument('--job_identifier', type=str,
        help="Slurm job ID, or other identifier, to not overwrite output.")
    parser.add_argument('--resume',type = int, default = 0, help="epoch num to resume from")
    return parser.parse_args(argv)
        
    
if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    # print(args)
    train_subjects = re.split('/', args.train_subjects)
    test_subjects = re.split('/',args.test_subjects)
    # print (args.config_file)
    config_dict_module = rhodin_utils_io.loadModule(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict['job_identifier'] = args.job_identifier
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    config_dict['data_dir_path'] = args.dataset_path
    config_dict['dataset_folder_train'] = args.dataset_path
    config_dict['dataset_folder_test'] = args.dataset_path
    root = args.dataset_path.rsplit('/', 2)[0]
    print (config_dict['test_subjects'])
    ignite = IgniteTrainNVS(config_dict_module.__file__, config_dict,args.resume)
    # if ignite is not None:
    ignite.run()
    # config_dict_module.__file__, config_dict)

