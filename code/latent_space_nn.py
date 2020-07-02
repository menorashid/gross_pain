import os
from helpers import util, visualize
import numpy as np
import pandas as pd
import numpy as np
import imageio
import torch
import sklearn.manifold
import sklearn.preprocessing

import train_encode_decode as ted
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import io as rhodin_utils_io

from train_encode_decode_pain import get_model_path 
import glob
# as ext_get_model_path

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

class IgniteTestNVS(ted.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
        
        data_loader = self.load_data_test(config_dict, ted.get_parameter_description(config_dict))
        model = self.load_network(config_dict)
        model = model.to(device)

        return model, data_loader, config_dict
    
def predict(model, input_dict, label_dict):
    model.eval()
    with torch.no_grad():
        input_dict_cuda, label_dict_cuda = rhodin_utils_datasets.nestedDictToDevice((input_dict, label_dict), device=device)
        output_dict_cuda = model(input_dict_cuda)
        output_dict = rhodin_utils_datasets.nestedDictToDevice(output_dict_cuda, device='cpu')
    return output_dict

def nextImage(data_iterator):
    input_dict, label_dict = next(data_iterator)
    return input_dict, label_dict

def get_config_model_and_iterator(config_path, config_dict):
    ignite = IgniteTestNVS()
    model, data_loader, config_dict = ignite.run(config_path, config_dict)
    data_iterator = iter(data_loader)
    return config_dict, model, data_iterator

def get_values(model, data_iterator, input_to_get, output_to_get):
    
    the_rest = {}
    for str_curr in input_to_get+output_to_get:
        the_rest[str_curr] = []
    
    idx = 0
    for input_dict, label_dict in data_iterator:
        idx+=1
        output_dict = predict(model, input_dict, label_dict)
        for str_curr in input_to_get:
            the_rest[str_curr].append(input_dict[str_curr].numpy())

        for str_curr in output_to_get:
            the_rest[str_curr].append(output_dict[str_curr].numpy())


    return the_rest


def set_up_config_dict(config_path,
                     train_subjects,
                     test_subjects,
                     job_identifier,
                     batch_size_test,
                     dataset_path,
                     input_to_get, 
                     output_to_get):
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
    for val in input_to_get:
        if val not in config_dict['input_types']:
            config_dict['input_types'].append(val) 
    for val in output_to_get:
        if val not in config_dict['output_types']:
            config_dict['output_types'].append(val) 
    return config_dict

def save_all_features(config_dict, config_path, all_subjects, out_path_meta, input_to_get, output_to_get):
    
    for test_subject_curr in all_subjects:
        print (test_subject_curr, all_subjects)
        out_dir_data = os.path.join(out_path_meta,test_subject_curr)
        config_dict['test_subjects'] = [test_subject_curr]


        config_dict, model, data_iterator = get_config_model_and_iterator(config_path, config_dict)
        ret_vals = get_values(model, data_iterator, input_to_get, output_to_get)
        
        for idx_batch, batch in enumerate(ret_vals['img_path']):
            new_batch = []
            for row in batch:
                [interval_int_pre, interval_int_post, interval_ind, view, frame] = row
                interval = '%014d_%06d'%(interval_int_pre,interval_int_post)
                img_path = util.get_image_name(test_subject_curr, interval_ind, interval, view, frame, config_dict['data_dir_path'])
                assert os.path.exists(img_path)
                new_batch.append(img_path)
            new_batch = np.array(new_batch)
            ret_vals['img_path'][idx_batch] = new_batch

        for k in ret_vals.keys():
            for idx_batch, batch in enumerate(ret_vals[k]):
                out_file = os.path.join(out_dir_data,k+'_%06d.npy'%idx_batch)
                util.makedirs(os.path.split(out_file)[0])
                np.save(out_file, batch)

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


def main():

    dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
    config_path = 'configs/config_train_rotation_crop.py'
    job_identifier = 'withRotCrop'
    nth_frame = 1

    dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
    config_path = 'configs/config_train_rotation_bl.py'
    job_identifier = 'withRot'
    nth_frame = 10
    
    dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
    config_path = 'configs/config_train_rotation_crop_newCal.py'
    job_identifier = 'withRotCropNewCal'
    nth_frame = 10

    # dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
    # config_path = 'configs/config_train_rotation_newCal.py'
    # job_identifier = 'withRotNewCal'
    nth_frame = 100

    train_subjects = 'brava/herrera/inkasso/julia/kastanjett/naughty_but_nice/sir_holger'.split('/')
    test_subject = 'aslan'
    all_subjects = [test_subject]+train_subjects

    model_num = 50
    batch_size_test = 64
    output_to_get = ['latent_3d']
    input_to_get = ['img_path']

    config_dict = set_up_config_dict(config_path, train_subjects, [test_subject], job_identifier, batch_size_test, dataset_path, input_to_get, output_to_get)
    model_path = get_model_path(config_dict, str(model_num))
    print (model_path)
    
    config_dict['pretrained_network_path'] = model_path
    config_dict['every_nth_frame'] = nth_frame
    out_path_meta = model_path[:-4]+'_feats'
    util.mkdir(out_path_meta)

    # save_all_features(config_dict, config_path, all_subjects, out_path_meta, input_to_get, output_to_get)
    
    simple = False
    cam_num = 3
    type_match = 'withtrain'
    config_dict['test_subjects'] = [train_subjects[0]]
    # post_str = '_'.join([config_dict['test_subjects'][0],'withself',str(cam_num)])
    post_str = '_'.join([config_dict['test_subjects'][0],type_match,str(cam_num)])
    if simple:
        md_file = os.path.join(out_path_meta,'_'.join(['nn', 'simple', post_str])+'.md')
    else:
        md_file = os.path.join(out_path_meta,'_'.join(['nn', 'rot', post_str])+'.md')

    dataset = ted.IgniteTrainNVS().load_data_test(config_dict, ted.get_parameter_description(config_dict)).dataset
    

    feat_files, im_files = get_file_list(out_path_meta, test_subject)
    feat_file = feat_files[0]
    im_file = im_files[0]
    q_feat_curr = np.load(feat_file)
    q_im = np.load(im_file)
    q_im_org = q_im
    q_feat_curr_org = q_feat_curr
    
    if not simple:
        q_im, q_feat_curr = get_rotated_latent(q_im, q_feat_curr, cam_num, dataset, test_subject)


    test_data = np.reshape(q_feat_curr, (q_feat_curr.shape[0],-1))
    
    if type_match=='withtrain':
        keep_lists = get_train_feat(train_subjects, out_path_meta, 1)
        train_data = np.reshape(keep_lists[0], (keep_lists[0].shape[0],-1))
        train_im = keep_lists[1]
    elif type_match=='withselforg':
        train_data = np.reshape(q_feat_curr_org, (q_feat_curr_org.shape[0],-1))
        train_im = q_im_org
    elif type_match=='withself':
        train_data = test_data
        train_im = q_im

    scaler = sklearn.preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    print (train_data.shape, test_data.shape)

    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=10, algorithm = 'brute')
    nn.fit(train_data)
    if type_match=='withself':
        _, idx_close = nn.kneighbors()
    else:
        _, idx_close = nn.kneighbors(test_data)
    print (idx_close[:10])
    # test_data)
    im_files = []
    captions =[]
    titles = ['query']+[str(num+1) for num in range(10)]

    for r in range(idx_close.shape[0]):
        val = q_im[r].replace('..','../../../../../..')
        title_str = os.path.split(q_im[r])[1].split('_')[-2]+' '+os.path.split(q_im[r])[1].split('_')[-1][:-4]
        title_strs = [title_str] 
        im_row = [val]
        for idx_curr in idx_close[r]:
            title_str = os.path.split(train_im[idx_curr])[1].split('_')[-2]+' '+os.path.split(train_im[idx_curr])[1].split('_')[-1][:-4]
            title_strs.append(title_str)
            im_row.append(train_im[idx_curr].replace('..','../../../../../..') )
        captions.append(title_strs)
        im_files.append(im_row)

    visualize.markdown_im_table(im_files, titles, feat_file, md_file, title_strs = captions)
    print (md_file)

    
        


    

if __name__=='__main__':
    main()