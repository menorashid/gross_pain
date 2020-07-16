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
    config_dict['bg_folder'] = '../data/median_bg/'
    config_dict['rot_folder'] = '../data/rotation_cal_1/'
    
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
    task = 'simple_featsave'
    
    edit_config_retvals(config_dict, input_to_get, output_to_get)

    for test_subject_curr in all_subjects:
        print (test_subject_curr, all_subjects)
        out_dir_data = os.path.join(out_path_meta,test_subject_curr)
        config_dict['test_subjects'] = [test_subject_curr]


        tester = IgniteTestNVS(config_path, config_dict, task)
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

def save_all_im(config_dict, config_path, all_subjects, out_path_meta):
    # (config_dict, config_path, all_subjects, out_path_meta, input_to_get, output_to_get, task):
    output_to_get = ['img_crop']
    input_to_get = ['img_crop']
    task = 'simple_imsave'
    
    edit_config_retvals(config_dict, input_to_get, output_to_get)

    for test_subject_curr in all_subjects:
        print (test_subject_curr, all_subjects)
        out_dir_data = os.path.join(out_path_meta,test_subject_curr)
        config_dict['test_subjects'] = [test_subject_curr]


        tester = IgniteTestNVS(config_path, config_dict, task)
        ret_vals = tester.get_images(input_to_get, output_to_get)
        
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


def get_file_list(out_path_meta, test_subject):
    dir_curr = os.path.join(out_path_meta, test_subject)
    feat_files = [os.path.join(dir_curr,file_curr) for file_curr in os.listdir(dir_curr) if file_curr.startswith('latent_3d')]
    feat_files.sort()
    im_files = [feat_file.replace('latent_3d','img_path') for feat_file in feat_files]
    for im_file in im_files:
        assert os.path.exists(im_file)
    return feat_files, im_files

def get_train_feat(train_subjects, meta_path, every_nth):
    all_feat_files = []
    all_im_files = []
    for subject in train_subjects:
        feat_files, im_files = get_file_list(meta_path, subject)
        all_feat_files += feat_files
        all_im_files += im_files

    keep_lists =[[],[]]
    for feat_file, im_file in zip(all_feat_files, all_im_files):
        for idx_file_curr, file_curr in enumerate([feat_file, im_file]):
            feat = np.load(file_curr)
            feat = feat[::every_nth]
            keep_lists[idx_file_curr].append(feat)

    
    keep_lists = [np.concatenate(keep_list, axis = 0) for keep_list in keep_lists]

    return keep_lists

def get_rotated_latent(q_im, q_feat_curr, cam_num, dataset, test_subject):

    cam2world_s = []
    for idx_q in range(q_im.shape[0]):
        im_file_curr = q_im[idx_q]
        im_file_curr_split = im_file_curr.split('/')
        view = int(im_file_curr_split[-2])
        subject = im_file_curr_split[-4]
        assert subject==test_subject
        rot_path = dataset.get_rot_path( view, subject, 'extrinsic_rot_inv')
        cam2world_s.append(np.load(rot_path))

    cam2world_s = np.array(cam2world_s)
    if cam_num>=0:
        world2cam_t = np.load(dataset.get_rot_path( cam_num, test_subject, 'extrinsic_rot'))[np.newaxis,:,:]
        cam2cam = np.matmul(world2cam_t,cam2world_s)
    else:
        cam2cam = cam2world_s

    cam2cam_t = np.transpose(cam2cam,(0,2,1))
    feat_new = np.matmul(q_feat_curr, cam2cam_t)
    
    return q_im, feat_new

def get_job_params(job_identifier, out_path_postpend, test_subjects = None, train_subjects = None, model_num = 50, batch_size_test = 64, test_every = None):
    if job_identifier=='withRotCrop':
        dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
        config_path = 'configs/config_train_rotation_crop.py'
        nth_frame = 1
    else:
        raise ValueError('job_identifier %s not registered'%job_identifier)

    all_subjects = 'aslan/brava/herrera/inkasso/julia/kastanjett/naughty_but_nice/sir_holger'.split('/')
    if test_subjects is None:
        test_subjects = ['aslan']
    if train_subjects is None:
        train_subjects = [val for val in all_subjects if val not in test_subjects]
    
    # model_num = 50
    # batch_size_test = 64
    # if 'featsave' in task:
    #     output_to_get = ['latent_3d']
    #     input_to_get = ['img_path']
    # elif 'imsave' in task:
    #     output_to_get = ['img_crop']
    #     input_to_get = ['img_crop']
    # else:
    #     raise ValueError('task %s not registered'%task)

    
    config_dict = set_up_config_dict(config_path, train_subjects, test_subjects, job_identifier, batch_size_test, dataset_path)
    model_path = get_model_path(config_dict, str(model_num))
    print (model_path)
    
    if test_every is None:
        test_every = nth_frame

    config_dict['pretrained_network_path'] = model_path
    config_dict['every_nth_frame'] = test_every
    out_path_meta = model_path[:-4]+'_'+out_path_postpend+'_'+str(test_every)
    util.mkdir(out_path_meta)
    params = {'config_dict':config_dict, 'config_path':config_path, 'all_subjects':all_subjects, 'out_path_meta':out_path_meta}
    return params

def main():

    job_identifier = 'withRotCrop'
    test_every = 100
    task = 'simple_imsave'
    job_params = get_job_params(job_identifier, task, test_every = test_every)
    if 'featsave' in task:
        save_all_features(**job_params)
    elif 'imsave' in task:
        save_all_im(**job_params)
    


if __name__=='__main__':
    main()