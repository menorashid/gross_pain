import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rhodin.python.utils import io as rhodin_utils_io
import torchvision
import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import torch.optim
import torch.nn

import train_encode_decode
import models.cnn_classifier as cnn_classifier
from multiview_dataset import MultiViewDataset
from rhodin.python.losses import generic as losses_generic
from rhodin.python.utils import datasets as rhodin_utils_datasets

class IgniteTrainPain(train_encode_decode.IgniteTrainNVS):
    def loadOptimizer(self, network, config_dict):
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
        loss = losses_generic.LossOnDict(key='pain', loss=loss_function)
        
        losses_train = []
        losses_test = []
        
        if 'pain' in config_dict['output_types']:
            losses_train.append(loss)
            losses_test.append(loss)
                
        loss_train = losses_generic.PreApplyCriterionListDict(losses_train, sum_losses=True)
        loss_test  = losses_generic.PreApplyCriterionListDict(losses_test,  sum_losses=True)
                
        # annotation and pred is organized as a list, to facilitate multiple
        # output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self, config_dict):#, config_dict):
        shorter_train_subjects = [subject[:2] for subject in config_dict['train_subjects']]
        shorter_test_subjects = [subject[:2] for subject in config_dict['test_subjects']]
        folder = "../output/trainNVSPain_{job_identifier}_{which_cnn}_pretr{pretrained_cnn}/{training_set}/nth{every_nth_frame}_c{active_cameras}_train{}_test{}_bs{use_view_batches}_lr{learning_rate}".format(shorter_train_subjects, shorter_test_subjects,**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
        return folder

    def load_data_train(self,config_dict):
        dataset = MultiViewDataset(data_folder=config_dict['dataset_folder_train'],
                                   bg_folder=config_dict['bg_folder'],
                                   input_types=config_dict['input_types'],
                                   label_types=config_dict['label_types_train'],
                                   subjects=config_dict['train_subjects'],
                                   rot_folder = config_dict['rot_folder'])

        loader = torch.utils.data.DataLoader(dataset, batch_size=config_dict['batch_size_train'],
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=0, pin_memory=False,
                                             collate_fn=rhodin_utils_datasets.default_collate_with_string)
        return loader
    
    def load_data_test(self,config_dict):
        dataset = MultiViewDataset(data_folder=config_dict['dataset_folder_test'],
                                   bg_folder=config_dict['bg_folder'],
                                   input_types=config_dict['input_types'],
                                   label_types=config_dict['label_types_test'],
                                   subjects=config_dict['test_subjects'],
                                   rot_folder = config_dict['rot_folder'])

        loader = torch.utils.data.DataLoader(dataset, batch_size=config_dict['batch_size_train'],
                                             shuffle=False,
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

    ignite = IgniteTrainPose()
    ignite.run(config_dict_module.__file__, config_dict)

