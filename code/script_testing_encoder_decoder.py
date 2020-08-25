import os
from helpers import util, visualize
import numpy as np
import pandas as pd
import numpy as np
import imageio
import torch
import sklearn.manifold
import sklearn.preprocessing

from test_encode_decode_new import IgniteTestNVS
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import io as rhodin_utils_io

from train_encode_decode_pain import get_model_path 
import glob
import imageio

# def get_rot_mats(config_dict, test_subject, view):
#     rot_folder = config_dict['rot_folder']


def set_up_config_dict(config_path,
                     train_subjects,
                     test_subjects,
                     job_identifier,
                     batch_size_test,
                     dataset_path):
                     # input_to_get, 
                     # output_to_get):
    config_dict = rhodin_utils_io.loadModule(config_path).config_dict
    config_dict['job_identifier'] = job_identifier
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    config_dict['implicit_rotation'] = config_dict.get('implicit_rotation', False)
    config_dict['skip_background'] = config_dict.get('skip_background', True)
    config_dict['loss_weight_pose3D'] = config_dict.get('loss_weight_pose3D', 0)
    config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
    config_dict['batch_size_test'] = batch_size_test
    config_dict['data_dir_path'] = dataset_path
    config_dict['dataset_folder_train'] = dataset_path
    config_dict['dataset_folder_test'] = dataset_path
    # config_dict['bg_folder'] = '../data/median_bg/'
    # config_dict['rot_folder'] = '../data/rotation_cal_2/'
    
    return config_dict

def edit_config_retvals(config_dict, input_to_get, output_to_get):
    for val in input_to_get:
        if val not in config_dict['input_types']:
            config_dict['input_types'].append(val) 
    for val in output_to_get:
        if val not in config_dict['output_types']:
            config_dict['output_types'].append(val) 

def save_all_features(config_dict, config_path, all_subjects, out_path_meta):
    output_to_get = ['latent_3d']
    input_to_get = ['img_path']
    # task = 'simple_featsave'
    
    edit_config_retvals(config_dict, input_to_get, output_to_get)

    for test_subject_curr in all_subjects:
        print (test_subject_curr, all_subjects)
        out_dir_data = os.path.join(out_path_meta,test_subject_curr)
        config_dict['test_subjects'] = [test_subject_curr]


        tester = IgniteTestNVS(config_path, config_dict)
        ret_vals = tester.get_values(input_to_get, output_to_get)
        
        for idx_batch, batch in enumerate(ret_vals[0]['img_path']):
            new_batch = []
            for row in batch:
                [interval_int_pre, interval_int_post, interval_ind, view, frame] = row
                interval = '%014d_%06d'%(interval_int_pre,interval_int_post)
                img_path = util.get_image_name(test_subject_curr, interval_ind, interval, view, frame, config_dict['data_dir_path'])
                assert os.path.exists(img_path)
                new_batch.append(img_path)
            new_batch = np.array(new_batch)
            ret_vals[0]['img_path'][idx_batch] = new_batch

        for ret_vals_inner in ret_vals:
            for k in ret_vals_inner.keys():
                for idx_batch, batch in enumerate(ret_vals_inner[k]):
                    out_file = os.path.join(out_dir_data,k+'_%06d.npy'%idx_batch)
                    util.makedirs(os.path.split(out_file)[0])
                    np.save(out_file, batch)

def save_all_im(config_dict, config_path, all_subjects, out_path_meta, view = None, bg = None):
    # (config_dict, config_path, all_subjects, out_path_meta, input_to_get, output_to_get, task):
    output_to_get = ['img_crop']
    input_to_get = ['img_crop']
    # task = 'simple_imsave'
    
    edit_config_retvals(config_dict, input_to_get, output_to_get)

    for test_subject_curr in all_subjects:
        print (test_subject_curr, all_subjects)
        out_dir_data = os.path.join(out_path_meta,test_subject_curr)
        config_dict['test_subjects'] = [test_subject_curr]


        tester = IgniteTestNVS(config_path, config_dict)
        ret_vals = tester.get_images(input_to_get, output_to_get, view, bg)
        
        for idx_batch, in_batch in enumerate(ret_vals[0]['img_crop']):
            out_batch = ret_vals[1]['img_crop'][idx_batch]
            for im_num in range(in_batch.shape[0]):
                out_file_pre = os.path.join(out_dir_data, '%04d_%04d')%(idx_batch,im_num)
                util.makedirs(out_dir_data)

                out_file = out_file_pre+'_in.jpg'
                imageio.imsave( out_file, in_batch[im_num])
                out_file = out_file_pre+'_out.jpg'
                imageio.imsave( out_file, out_batch[im_num])

        visualize.writeHTMLForFolder(out_dir_data, height = 128, width = 128)
        
def save_im_rot(config_dict, config_path, all_subjects, out_path_meta):
    output_to_get = ['img_crop']
    input_to_get = ['img_crop']
    edit_config_retvals(config_dict, input_to_get, output_to_get)
    config_dict['batch_size_test'] = 4
    
    test_subject_curr = all_subjects[0]
    
    out_dir_data = os.path.join(out_path_meta,test_subject_curr)
    config_dict['test_subjects'] = [test_subject_curr]

    tester = IgniteTestNVS(config_path, config_dict)
    ret_vals = tester.rotate_one_image()
        
    for im_num,im in enumerate(ret_vals):
        out_file_pre = os.path.join(out_dir_data, '%04d')%(im_num)
        util.makedirs(out_dir_data)

        out_file = out_file_pre+'.jpg'
        imageio.imsave( out_file, im)
        
    visualize.writeHTMLForFolder(out_dir_data, height = 128, width = 128)

def save_latent_view_diff(config_dict, config_path, all_subjects, out_path_meta, views = [0,1,2,3]):
    # (config_dict, config_path, all_subjects, out_path_meta, input_to_get, output_to_get, task):
    output_to_get = ['latent_3d']
    input_to_get = ['img_path','view','frame']
    # task = 'simple_imsave'
    
    edit_config_retvals(config_dict, input_to_get, output_to_get)

    for test_subject_curr in all_subjects:
        out_dir_data = os.path.join(out_path_meta,test_subject_curr)
        config_dict['test_subjects'] = [test_subject_curr]

        tester = IgniteTestNVS(config_path, config_dict)
        ret_vals = tester.get_latent_diff(views)
        
        for k in ret_vals.keys():
            vals = ret_vals[k]
        
        for idx_batch in range(len(ret_vals['diffs'])):
            out_file = os.path.join(out_dir_data, '%04d.npz')%(idx_batch)
            util.makedirs(out_dir_data)
            print (np.mean(ret_vals['diffs'][idx_batch]))
            inner_batch = {}
            for k in ret_vals.keys():
                inner_batch[k] = ret_vals[k][idx_batch]
            np.savez_compressed(out_file, **inner_batch)
        

def get_dataset_path(job_identifier):
    if 'flowcroppercent' in job_identifier.lower():
        dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/'
    elif 'flowcrop' in job_identifier.lower():
        dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/'
    elif 'crop' in job_identifier.lower():
        dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
    else:
        dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
    return dataset_path

def get_config_path(job_identifier):
    if job_identifier=='withRotCrop':
        config_path = 'configs/config_train_rotation_crop.py'
    elif job_identifier=='withRotCropNewCal':
        config_path = 'configs/config_train_rotation_crop_newCal.py'
    elif job_identifier=='withRotNewCal':
        config_path = 'configs/config_train_rotation_newCal.py'
    elif job_identifier=='withRotTransAll':
        config_path = 'configs/config_train_rotation_translation_newCal.py'
    elif job_identifier=='withRotSeg':
        config_path = 'configs/config_train_rot_segmask.py'
    elif job_identifier=='withRotCropSeg':
        config_path = 'configs/config_train_rotCrop_segmask.py'
    elif job_identifier=='withRotTranslateSeg':
        config_path = 'configs/config_train_rotTranslate_segmask.py'
    elif job_identifier=='withRotCropSegLatent': 
        config_path = 'configs/config_train_rotCropSegMaskLatent.py'
    elif job_identifier=='withRotCropLatent': 
        config_path = 'configs/config_train_rotCropLatent.py'
    # elif job_identifier=='withRotFlowCropLatent': 
    #     config_path = 'configs/config_train_rotFlowCropLatent.py'
    elif job_identifier=='withRotFlowCropLatentPercentLatentLr0.1':
        config_path = 'configs/config_train_rotFlowCropLatent.py'
    elif job_identifier=='withRotFlowCropPercentBetterBg':
        config_path = 'configs/config_train_rotFlowCropBetterBg.py'
    elif job_identifier=='withRotFlowCropPercent':
        config_path = 'configs/config_train_rotFlowCrop.py'
    else:
        raise ValueError('job_identifier %s not registered'%job_identifier)

    return config_path

def get_subjects(train_subjects, test_subjects):
    all_subjects = 'aslan/brava/herrera/inkasso/julia/kastanjett/naughty_but_nice/sir_holger'.split('/')
    if test_subjects is None:
        test_subjects = ['aslan']
    
    if train_subjects is None:
        train_subjects = [val for val in all_subjects if val not in test_subjects]
    elif train_subjects=='all':
        train_subjects = all_subjects
    return train_subjects, test_subjects, all_subjects

def get_job_params(job_identifier, out_path_postpend, test_subjects = None, train_subjects = None, model_num = 50, batch_size_test = 64, test_every = None):
    
    dataset_path = get_dataset_path(job_identifier)
    config_path = get_config_path(job_identifier)
    
    all_subjects = 'aslan/brava/herrera/inkasso/julia/kastanjett/naughty_but_nice/sir_holger'.split('/')
    
    train_subjects, test_subjects, all_subjects = get_subjects(train_subjects, test_subjects)

    # if test_subjects is None:
    #     test_subjects = ['aslan']
    
    # if train_subjects is None:
    #     train_subjects = [val for val in all_subjects if val not in test_subjects]
    # elif train_subjects=='all':
    #     train_subjects = all_subjects

       
    config_dict = set_up_config_dict(config_path, train_subjects, test_subjects, job_identifier, batch_size_test, dataset_path)
    model_path = get_model_path(config_dict, str(model_num))
    if not os.path.exists(model_path):
        model_path_old = get_model_path(config_dict, str(model_num), True)
        if os.path.exists(model_path_old):
            model_path_meta = os.path.split(os.path.split(model_path)[0])[0]
            model_old_path_meta = os.path.split(os.path.split(model_path_old)[0])[0]
            os.rename(model_old_path_meta, model_path_meta)
    
    print (model_path)
    if not os.path.exists(model_path):
        print ('model path does not exist')
        return None
    
    
    # if test_every is None:
    #     test_every = nth_frame

    config_dict['pretrained_network_path'] = model_path
    config_dict['every_nth_frame'] = test_every
    out_path_meta = model_path[:-4]+'_'+out_path_postpend+'_'+str(test_every)
    util.mkdir(out_path_meta)
    params = {'config_dict':config_dict, 'config_path':config_path, 'all_subjects':all_subjects, 'out_path_meta':out_path_meta}
    return params

def main():

    job_identifier = 'withRotCropNewCal'
    # job_identifier = 'withRotNewCal'
    # job_identifier = 'withRotTransAll'
    # job_identifier = 'withRotSeg'
    # job_identifier = 'withRotCropSeg'
    # job_identifier = 'withRotTranslateSeg'
    # job_identifier = 'withRotCropSegLatent'
    
    # job_identifier = 'withRotCropLatent'
    # train_subjects = 'all'

    # job_identifier = 'withRotFlowCropLatentPercentLatentLr0.1'
    # job_identifier = 'withRotFlowCropPercentBetterBg'
    # job_identifier = 'withRotFlowCropPercent'


    bg = None
    # '../data/blank_mean.jpg'
    train_subjects = None
    # test_subjects ='aslan'

    if 'crop' in job_identifier.lower():
        test_every = 1000
    else:
        test_every = 1000

    # task = 'imsave'
    # task = 'bgswap'
    # task = 'viewdiff'
    task = 'imrot'
    view = None
    job_params = get_job_params(job_identifier, task, train_subjects = train_subjects, test_every = test_every)
    
    if job_params is None:
        return
    

    if 'featsave' in task:
        save_all_features(**job_params)
    elif ('imsave' in task) or ('bgswap' in task):
        save_all_im(**job_params, view = view, bg = bg)
    elif 'viewdiff' in task:
        save_latent_view_diff(**job_params)
    elif 'imrot' in task:
        save_im_rot(**job_params)
    


if __name__=='__main__':
    main()