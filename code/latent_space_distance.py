import os
from helpers import util, visualize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision
import importlib
import imageio
# from tsnecuda import TSNE
import torch
import sklearn.manifold
import sklearn.preprocessing

import train_encode_decode
import multiview_dataset
from rhodin.python.utils import datasets as rhodin_utils_datasets

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

class IgniteTestNVS(train_encode_decode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)

        data_loader = self.load_data_test(config_dict)

        # load model
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

def get_config_model_and_iterator(config_path, train_subjects, test_subjects, batch_size_test, dataset_path, model_path):
    config_dict_module = multiview_dataset.rhodin_utils_io.loadModule(config_path)
    config_dict = config_dict_module.config_dict
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    config_dict['batch_size_test'] = batch_size_test
    config_dict['data_dir_path'] = dataset_path
    config_dict['dataset_folder_train'] = dataset_path
    config_dict['dataset_folder_test'] = dataset_path
    config_dict['pretrained_network_path'] = model_path

    ignite = IgniteTestNVS()
    model, data_loader, config_dict = ignite.run(config_dict_module.__file__, config_dict)
    data_iterator = iter(data_loader)
    return config_dict, model, data_iterator

def get_labels_for_whole_test_set(model, data_iterator, config_dict, str_to_get_output = None):
    # print (len(data_iterator))
    # print (config_dict['batch_size_test'])
    # iterations = int(len(data_iterator)/config_dict['batch_size_test'])
    pains = []
    views = []
    latent_3d = []
    if str_to_get_output is not None:
        the_rest = {}
        for str_curr in str_to_get_output:
            the_rest[str_curr] = []
    # latent_3d_rotated = []
    # shuffled_pose = []
    # with tqdm(total=iterations) as pbar:
    #     for i in range(iterations):
    #         pbar.update(1)
    idx = 0
    for input_dict, label_dict in data_iterator:
        idx+=1
        output_dict = predict(model, input_dict, label_dict)
        pains.append(label_dict['pain'])
        views.append(label_dict['view'])
        latent_3d.append(output_dict['latent_3d'])
        if str_to_get_output is not None:
            for str_curr in str_to_get_output:
                the_rest[str_curr].append(output_dict[str_curr])

    if str_to_get_output is not None:
        return pains, views, latent_3d, the_rest
    else:
        return pains, views, latent_3d


def prepare_data_for_TSNE(pains, views, latent_3d):
    X = np.concatenate(latent_3d)
    X = np.reshape(X, (X.shape[0], -1))
    pains = np.concatenate(pains)
    views = np.concatenate(views)
    return pains, views, X

def prepare_data_for_camera_rot(pains, views, latent_3d):
    X = np.concatenate(latent_3d)
    # X = np.reshape(X, (X.shape[0], -1))
    pains = np.concatenate(pains)
    views = np.concatenate(views)
    return pains, views, X


def plot_TSNE(X_embedded, labels,out_file):
    plt.figure()
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels)
    plt.savefig(out_file)
    plt.close()


def dot_dists(X_rel):
    
    X_rel = X_rel/np.linalg.norm(X_rel, axis = 1, keepdims = True)
    # print (X_rel.shape, X_rel.T.shape)
    dots = np.matmul(X_rel, X_rel.T)
    dots = dots[np.triu_indices(dots.shape[0], k=1)]
    return dots

def scratch_dump_whatevs():
    test_subject = 'aslan'
    out_dir  = '../scratch/dist_hists'
    out_file = os.path.join(out_dir, 'info.npz')
    cam_dir = '../data/rotation_cal_1/'
    rot_str = 'extrinsic_rot_'
    rot_inv_str = 'extrinsic_rot_inv_'

    viewpoints = pd.read_csv('../metadata/viewpoints.csv',index_col = 'subject')

    viewpoints = viewpoints.loc[test_subject,:].values

    rots = [np.load(os.path.join(cam_dir,rot_str+str(val)+'.npy')) for val in viewpoints]
    rot_invs = [np.load(os.path.join(cam_dir,rot_inv_str+str(val)+'.npy')) for val in viewpoints]

    loaded_data = np.load(out_file)
    [pains, views, X] = [loaded_data[val] for val in ['pains','views','X']]

    X_t_all = []
    views_new_all = []
    for view in range(4):
        bin_view = views == view
        
        X_rel = X[bin_view,:,:]
        
        rot_inv = rot_invs[view][np.newaxis,:,:]
        rot_inv = np.tile(rot_inv, (X_rel.shape[0],1,1))

        X_rel = np.transpose(X_rel,(0,2,1))
        
        X_t = np.matmul(rot_inv, X_rel)
        # X_t = X_rel

        X_t = np.transpose(X_t,(0,2,1))
        # print (X_t.shape, X_t[:1,:10])

        X_t_all.append(X_t)
        views_new = np.ones((len(X_t),))*view
        views_new_all.append(views_new)
    
    diffs = []
    for view in range(4):
        X_t_1 = X_t_all[view]
        for view_2 in range(view+1, 4):
            X_t_2 = X_t_all[view_2]
            diff_curr = np.sqrt(np.sum(np.power(X_t_1 - X_t_2,2),axis = 2))
            # mean_per_point = np.mean(diff_curr, axis =1)
            diffs.append(diff_curr)

    diffs = np.array(diffs)
    print (diffs.shape)

    print (np.mean(np.mean(diffs,axis = 2), axis = 1))
    print (np.mean(np.mean(diffs,axis = 2), axis = 0))
    print (np.mean(np.mean(diffs,axis = 0), axis = 0))

    X_t_all = np.concatenate(X_t_all, axis = 0)
    views_new = np.concatenate(views_new_all, axis = 0)
    print (views_new[::10])
    print (X_t_all.shape, views_new.shape)
    X_t_all = np.reshape(X_t_all,(X_t_all.shape[0],X_t_all.shape[1]*X_t_all.shape[2]))
    X_t_all = sklearn.preprocessing.StandardScaler().fit_transform(X_t_all)

    tsne = sklearn.manifold.TSNE(perplexity = 10)
    X_embedded = tsne.fit_transform(X_t_all)

    # # Plot with relevant labels
    plot_TSNE(X_embedded, views_new, os.path.join(out_dir, 'world_view_tsne.jpg'))
    # 

    # print (len(diffs),diffs)
        

def script_save_npz():
    meta_path = '../output/trainNVS_withRotCrop_UNet_layers4_implRFalse_w3Dp0_w3D0_wRGB1_wGrad0o01_wImgNet2/skipBGTrue_bg0_fg24_3d600_lh3Dp2_ldrop0o3_billinupper_fscale4_shuffleFGTrue_shuffle3dTrue_LPS_2fps_crop/nth1_cFalse_train{}_test{}_bs4_lr0o001'

    dataset_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'

    train_subjects = 'brava/herrera/inkasso/julia/kastanjett/naughty_but_nice/sir_holger'
    test_subjects = 'aslan'

    config_path = 'configs/config_test_debug.py'
    str_to_get_output = ['latent_3d_rotated','shuffled_pose']

    tsne = sklearn.manifold.TSNE(perplexity = 4)
    # TSNE()
    batch_size_test = 4
    model_num = 50

    train_subjects = train_subjects.split('/')
    test_subjects = test_subjects.split('/')
    
    train_subjects_str = [val[:2] for val in train_subjects]
    test_subjects_str = [val[:2] for val in test_subjects]
    meta_path = meta_path.format(train_subjects_str,test_subjects_str).replace(',','_').replace(' ','')
    print (meta_path)
    
    model_path = os.path.join(meta_path, 'models','network_%03d.pth'%model_num)
    
    

    # Load model and data
    config_dict, model, data_iterator = get_config_model_and_iterator(config_path,
                                                                      train_subjects,
                                                                      test_subjects,
                                                                      batch_size_test, 
                                                                      dataset_path,
                                                                      model_path)

    if str_to_get_output is not None:
        pains, views, latent_3d, the_rest = get_labels_for_whole_test_set(model, data_iterator, config_dict, str_to_get_output)

        for k in the_rest:
            the_rest[k] = np.concatenate(the_rest[k])
            print (k, the_rest[k].shape)
    else:
        pains, views, latent_3d = get_labels_for_whole_test_set(model, data_iterator, config_dict, str_to_get_output)        
    
    print (views)

    pains, views, X = prepare_data_for_camera_rot(pains, views, latent_3d)
    print (pains.shape, views.shape, X.shape)

    out_dir  = '../scratch/dist_hists'
    out_file = os.path.join(out_dir, 'info_with_rest.npz')
    np.savez(out_file, pains = pains, views = views, X = X, **the_rest)

        
def transform_to_world(X, views, rots, rot_invs):
    X_new = np.zeros(X.shape)
    for view in np.unique(views):
        bin_view = views == view

        X_rel = X[bin_view,:,:]

        rot_inv = rot_invs[view]
        rot_inv = rot_inv[np.newaxis,:,:]
        rot_inv = np.tile(rot_inv, (X_rel.shape[0],1,1))

        X_rel = np.transpose(X_rel,(0,2,1))
        X_t = np.matmul( rot_inv, X_rel)
        X_t = np.transpose(X_t,(0,2,1))

        X_new[bin_view,:,:] = X_t
        
    return X_new

def transform_to_cam(X, views, rots, rot_invs, views_list):
    X_in_cams = []
    for view in views_list:
        rot_list = []

        for view_pt in views:
            rot_curr = np.matmul(rots[view], rot_invs[view_pt])
            rot_list.append(rot_curr)
        rot_list = np.array(rot_list)

        X_in_cam = np.matmul(rot_list, np.transpose(X, (0,2,1)))
        X_in_cam = np.transpose(X_in_cam, (0,2,1))
        X_in_cams.append(X_in_cam)

    return X_in_cams


def save_time_dots_hist(X, out_file_curr):

    time_dots = []
    # X = X.reshape((-1,4,600))
    for idx_time in range(int(X.shape[0]/4)):
        time_feat = X[4*idx_time:4*idx_time+4]
        dots = dot_dists(time_feat)
        time_dots.append(dots)

    time_dots = np.concatenate(time_dots)
    print (time_dots.shape)
    title = 'Time'
    visualize.hist(time_dots,out_file_curr,bins=30,normed=False,xlabel='Value',ylabel='Frequency',title=title)


def main():
    # script_save_npz()
    test_subject = 'aslan'
    out_dir  = '../scratch/dist_hists'
    out_file = os.path.join(out_dir, 'info_with_rest.npz')
    cam_dir = '../data/rotation_cal_1/'
    rot_str = 'extrinsic_rot_'
    rot_inv_str = 'extrinsic_rot_inv_'
    views_list = [0,1,2,3]
    viewpoints = pd.read_csv('../metadata/viewpoints.csv',index_col = 'subject')

    viewpoints = viewpoints.loc[test_subject,:].values

    rots = [np.load(os.path.join(cam_dir,rot_str+str(val)+'.npy')) for val in viewpoints]
    rot_invs = [np.load(os.path.join(cam_dir,rot_inv_str+str(val)+'.npy')) for val in viewpoints]

    loaded_data = np.load(out_file)
    [pains, views, X, X_rot, shuf_idx] = [loaded_data[val] for val in ['pains','views','X']+['latent_3d_rotated','shuffled_pose']]

    # print (X[0,0,:],X_rot[0,0,:])
    # print (np.min(X-X_rot), np.max(X-X_rot))

    # print (views[:10])
    # print (shuf_idx)
    scaler = sklearn.preprocessing.StandardScaler()
    # tsne = sklearn.manifold.TSNE(perplexity = 30)
    
    # X_w = transform_to_world(X, views, rots, rot_invs)
    X_per_cam = transform_to_cam(X, views, rots, rot_invs, views_list)

    X_per_cam = [np.reshape(X_rel, (-1,4,200,3)) for X_rel in X_per_cam]
    X_rot = np.reshape(X_rot, (-1,4,200,3))
    shuf_idx = np.reshape(shuf_idx, (-1,4))
    views = np.reshape(views, (-1,4))

    for t in range(X_rot.shape[0]):
        print ('views[t]', views[t])
        print ('shuf_idx[t]', shuf_idx[t])
        print ('X_rot[t,:3]', X_rot[t,:,:3])
        for idx_x,x in enumerate(X_per_cam):
            print ('idx_x, x[t,:3]', idx_x, x[t,:,:3])
        break





    # X_per_cam = [np.reshape(X_rel, (X_rel.shape[0],-1)) for X_rel in X_per_cam]
    # X = np.reshape(X, (X.shape[0],-1))
    
    # out_file = os.path.join(out_dir,'new_hist_time.jpg')
    # save_time_dots_hist(scaler.fit_transform(X), out_file)
    # for idx_X_curr, X_curr in enumerate(X_per_cam):
    #     out_file = os.path.join(out_dir,'new_hist_time_view'+str(idx_X_curr)+'.jpg')
    #     save_time_dots_hist(scaler.fit_transform(X_curr), out_file)
    
    

    # pains, views, X = prepare_data_for_TSNE(pains, views, latent_3d)    


    # pains, views, X = prepare_data_for_TSNE(pains, views, latent_3d)
    
    # print (pains.shape, views.shape, X.shape)

    # out_dir = os.path.join('../scratch','dist_hists')
    # util.mkdir(out_dir)

    # for view in range(4):
    #     X_rel = X[views==view,:]
    #     dots = dot_dists(X_rel)
    #     out_file_curr = os.path.join(out_dir,'hist_view_'+str(view)+'.jpg')
    #     title = 'View '+str(view)
    #     visualize.hist(dots,out_file_curr,bins=30,normed=False,xlabel='Value',ylabel='Frequency',title=title)


    # time_dots = []
    # # X = X.reshape((-1,4,600))
    # for idx_time in range(int(X.shape[0]/4)):
    #     time_feat = X[4*idx_time:4*idx_time+4]
    #     dots = dot_dists(time_feat)
    #     time_dots.append(dots)

    # time_dots = np.concatenate(time_dots)
    # print (time_dots.shape)
    # out_file_curr = os.path.join(out_dir,'hist_time.jpg')
    # title = 'Time'
    # visualize.hist(time_dots,out_file_curr,bins=30,normed=False,xlabel='Value',ylabel='Frequency',title=title)


    # X = sklearn.preprocessing.StandardScaler().fit_transform(X)

    # X_embedded = tsne.fit_transform(X)

    # # Plot with relevant labels
    # plot_TSNE(X_embedded, pains, os.path.join(out_dir, 'pain_tsne.jpg'))
    # plot_TSNE(X_embedded, views, os.path.join(out_dir, 'views_tsne.jpg'))
    
    # visualize.writeHTMLForFolder(out_dir)


    

if __name__=='__main__':
    main()