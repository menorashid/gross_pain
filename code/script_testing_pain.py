import os
from helpers import util

import script_testing_encoder_decoder as sted
import train_encode_decode_pain as tedp
from rhodin.python.utils import io as rhodin_utils_io
from test_encode_decode_pain import IgniteTestPain

def set_up_config_dict(config_path, train_subjects, test_subjects, job_identifier, job_identifier_encdec, batch_size_test, dataset_path):
    config_dict_module = rhodin_utils_io.loadModule(config_path)
    config_dict = config_dict_module.config_dict
    config_dict['job_identifier_encdec'] = job_identifier_encdec
    config_dict['job_identifier'] = job_identifier
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    config_dict['batch_size_test'] = batch_size_test
    config_dict['data_dir_path'] = dataset_path
    config_dict['dataset_folder_train'] = dataset_path
    config_dict['dataset_folder_test'] = dataset_path

    config_dict['implicit_rotation'] = config_dict.get('implicit_rotation', False)
    config_dict['skip_background'] = config_dict.get('skip_background', True)
    config_dict['loss_weight_pose3D'] = config_dict.get('loss_weight_pose3D', 0)
    config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
    
    return config_dict
    

def get_job_params(job_identifier, job_identifier_encdec, out_path_postpend, test_subjects = None, train_subjects = None, model_num = 10, batch_size_test = 1024, test_every = None):
    
    dataset_path = sted.get_dataset_path(job_identifier)
    # print ('dataset_path',dataset_path)

    if job_identifier=='painWithRotCrop':
        config_path = 'configs/config_train_painfromlatent.py'
    elif job_identifier=='pain':
        config_path = 'configs/config_train_painfromlatent_crop.py'

    train_subjects, test_subjects, all_subjects = sted.get_subjects(train_subjects, test_subjects)

    config_dict = set_up_config_dict(config_path, train_subjects, test_subjects, job_identifier, job_identifier_encdec,batch_size_test, dataset_path)
    model_path = tedp.get_model_path_pain(config_dict, str(model_num))
    if not os.path.exists(model_path):
        model_path_old = tedp.get_model_path_pain(config_dict, str(model_num), True)
        if os.path.exists(model_path_old):
            model_path_meta = os.path.split(os.path.split(model_path)[0])[0]
            model_old_path_meta = os.path.split(os.path.split(model_path_old)[0])[0]
            os.rename(model_old_path_meta, model_path_meta)
    
    # print (model_path)
    if not os.path.exists(model_path):
        print ('model path does not exist', model_path)
        return None
    
    config_dict['pretrained_network_path'] = model_path
    config_dict['every_nth_frame'] = test_every
    out_path_meta = model_path[:-4]+'_'+out_path_postpend+'_'+str(test_every)
    util.mkdir(out_path_meta)
    params = {'config_dict':config_dict, 'config_path':config_path, 'all_subjects':all_subjects, 'out_path_meta':out_path_meta}
    return params


def main():
    job_identifier = 'painWithRotCrop'
    job_identifier_encdec = 'withRotCrop'

    job_identifier = 'pain'
    job_identifier_encdec = 'withRotCropNewCal'
    # job_identifier_encdec = 'withRotFlowCropPercent'
    # job_identifier_encdec = 'withRotFlowCropLatentPercentLatentLr0.1'
    # job_identifier_encdec = 'withRotFlowCropPercentBetterBg'


    train_subjects = None
    test_every = 1
    task = 'accuracy'
    job_params = get_job_params(job_identifier, job_identifier_encdec, task, train_subjects = train_subjects, test_every = test_every, test_subjects = ['brava'] )
    out_dir = job_params['out_path_meta']
    out_dir = os.path.split(os.path.split(out_dir)[0])[0]
    # print (out_dir)

    test_results = util.readLinesFromFile(os.path.join(out_dir, 'debug_log_testing.txt'))
    print (len(test_results), test_results[-1].split(',')[2])

    # config_dict = job_params['config_dict']

    # output_to_get = ['pain_pred']
    # input_to_get = ['pain']
    # sted.edit_config_retvals(config_dict, input_to_get, output_to_get)
    
    # out_dir_data = os.path.join(out_path_meta,test_subject_curr)
    # config_dict['test_subjects'] = [test_subject_curr]

    # tester = IgniteTestPain(job_params['config_path'], config_dict)
    # ret_vals = tester.get_accuracy()



if __name__=='__main__':
    main()