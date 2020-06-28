import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rhodin.python.utils import io as rhodin_utils_io
import torchvision
import torch
# torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import torch.optim
import torch.nn

import train_encode_decode
import models.cnn_classifier as cnn_classifier
from simple_frame_dataset import SimpleFrameDataset, SimpleRandomFrameSampler
from rhodin.python.losses import generic as losses_generic
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import training as rhodin_utils_train
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from helpers import util

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class IgniteTrainPain(train_encode_decode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
    
        # save path and config files
        save_path = self.get_parameter_description(config_dict)
        rhodin_utils_io.savePythonFile(config_dict_file, save_path)
        rhodin_utils_io.savePythonFile(__file__, save_path)
        
        # now do training stuff
        epochs = config_dict['num_epochs']
        train_loader = self.load_data_train(config_dict)
        test_loader = self.load_data_test(config_dict)
        model = self.load_network(config_dict)
        model = model.to(device)
        optimizer = self.load_optimizer(model,config_dict)
        loss_train, loss_test = self.load_loss(config_dict)
            
        trainer = create_supervised_trainer(model, optimizer, loss_train, device=device)
        evaluator = create_supervised_evaluator(model,
                                                metrics={'accuracy': Accuracy(),
                                                         'bceloss': Loss(loss_test)},
                                                device=device)
    
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_progress(engine):
            # log the loss
            iteration = engine.state.iteration - 1
            if iteration % config_dict['print_every'] == 0:
                util.save_training_error(save_path, engine)
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            _ = util.save_testing_error(save_path, trainer, evaluator,
                                dataset_str='Training Set',
                                save_extension='debug_log_training_wholeset.txt')
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(trainer.state.epoch, metrics['accuracy'], metrics['bceloss']))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            print("Test Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(trainer.state.epoch, metrics['accuracy'], metrics['bceloss']))

        @trainer.on(Events.ITERATION_COMPLETED)
        def validate_model(engine):
            iteration = engine.state.iteration - 1
            if (iteration+1) % config_dict['test_every'] != 0: # +1 to prevent evaluation at iteration 0
                return
            print("Running evaluation at iteration",iteration)
            evaluator.run(test_loader)
            avg_accuracy = util.save_testing_error(save_path, engine, evaluator,
                                dataset_str='Test Set', save_extension='debug_log_testing.txt')
            # save the best model
            rhodin_utils_train.save_model_state(save_path, trainer, avg_accuracy, model, optimizer, engine.state)
        
        @trainer.on(Events.EPOCH_COMPLETED)
        # @trainer.on(Events.ITERATION_COMPLETED)
        def save_model(engine):
            epoch = engine.state.epoch
            print ('epoch',epoch,'engine.state.iteration',engine.state.iteration)
            if not epoch % config_dict['save_every']: # +1 to prevent evaluation at iteration 0
                rhodin_utils_train.save_model_state_iter(save_path, trainer, model, optimizer, engine.state)

        # kick everything off
        trainer.run(train_loader, max_epochs=epochs)
        
    def load_optimizer(self, network, config_dict):
        # params_all_id = list(map(id, network.parameters()))


        # params_toOptimize = [p for p in network.parameters() ]

        # print("Static learning rate: {} params".format(len(params_static_id)))
        # print("Total: {} params".format(len(params_all_id)))

        opt_params = [{'params': network.parameters(), 'lr': config_dict['learning_rate']}]
        optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        return optimizer

    def load_network(self, config_dict):
        model = cnn_classifier.CNNClassifier(num_classes=config_dict['num_classes'],
                                            which_cnn=config_dict['which_cnn'],
                                            pretrained=config_dict['pretrained_cnn'],
                                            input_size=config_dict['inputDimension'],
                                            output_types=config_dict['output_types'])

        return model

    def load_loss(self, config_dict):

        # Binary cross-entropy for pain classification
        # loss_function = torch.nn.BCELoss()
        loss_function = torch.nn.CrossEntropyLoss()
        # loss = losses_generic.LossOnDict(key='pain', loss=loss_function)
        
        # losses_train = []
        # losses_test = []
        # 
        # if 'pain' in config_dict['output_types']:
        #     losses_train.append(loss)
        #     losses_test.append(loss)
                
        # loss_train = losses_generic.PreApplyCriterionListDict(losses_train, sum_losses=True)
        # loss_test  = losses_generic.PreApplyCriterionListDict(losses_test,  sum_losses=True)
                
        # annotation and pred is organized as a list, to facilitate multiple
        # output types (e.g. heatmap and 3d loss)
        return loss_function, loss_function

    def get_parameter_description(self, config_dict):#, config_dict):
        shorter_train_subjects = [subject[:2] for subject in config_dict['train_subjects']]
        shorter_test_subjects = [subject[:2] for subject in config_dict['test_subjects']]
        folder = "../output/trainNVSPain_{job_identifier}_{which_cnn}_pretr{pretrained_cnn}/{training_set}/nth{every_nth_frame}_c{active_cameras}_train{}_test{}_lr{learning_rate}_bstrain{batch_size_train}_bstest{batch_size_test}".format(shorter_train_subjects, shorter_test_subjects,**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
        return folder

    def load_data_train(self,config_dict):
        dataset = SimpleFrameDataset(data_folder=config_dict['dataset_folder_train'],
                                     subjects = config_dict['train_subjects'],
                                     input_types=config_dict['input_types'],
                                     label_types=config_dict['label_types_train'])

        sampler = SimpleRandomFrameSampler(
                  data_folder=config_dict['dataset_folder_train'],
                  subjects=config_dict['train_subjects'],
                  views=config_dict['views'],
                  every_nth_frame=config_dict['every_nth_frame'])
    
                                           
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config_dict['batch_size_train'],
                                             sampler=sampler,
                                             drop_last=True,
                                             num_workers=0, pin_memory=False,
                                             collate_fn=rhodin_utils_datasets.default_collate_with_string)
        return loader
    
    def load_data_test(self,config_dict):
        dataset = SimpleFrameDataset(data_folder=config_dict['dataset_folder_test'],
                                     subjects = config_dict['test_subjects'],
                                     input_types=config_dict['input_types'],
                                     label_types=config_dict['label_types_test'])

        sampler = SimpleRandomFrameSampler(
                  data_folder=config_dict['dataset_folder_test'],
                  subjects=config_dict['test_subjects'],
                  views=config_dict['views'],
                  every_nth_frame=config_dict['every_nth_frame'])
    
                                           
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config_dict['batch_size_test'],
                                             sampler=sampler,
                                             drop_last=True,
                                             num_workers=0, pin_memory=False,
                                             collate_fn=rhodin_utils_datasets.default_collate_with_string)

        return loader
    

if __name__ == "__main__":
    args = train_encode_decode.parse_arguments(sys.argv[1:])
    print(args)
    train_subjects = re.split('/', args.train_subjects)
    test_subjects = re.split('/',args.test_subjects)

    config_dict_module = rhodin_utils_io.loadModule(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict['job_identifier'] = args.job_identifier
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    config_dict['data_dir_path'] = args.dataset_path
    config_dict['dataset_folder_train'] = args.dataset_path
    config_dict['dataset_folder_test'] = args.dataset_path
    root = args.dataset_path.rsplit('/', 2)[0]
    config_dict['bg_folder'] = os.path.join(root, 'median_bg/')
    config_dict['rot_folder'] = os.path.join(root, 'rotation_cal_1/')

    ignite = IgniteTrainPain()
    ignite.run(config_dict_module.__file__, config_dict)

