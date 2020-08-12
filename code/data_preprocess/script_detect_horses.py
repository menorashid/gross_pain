import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
import pandas as pd
from helpers import util, visualize
from tqdm import tqdm

import cv2
import random

import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

import PIL
from PIL import Image

import multiprocessing

class HorseDetector():

    def __init__(self, data_path, 
                out_data_path, 
                thresh = 0.3, 
                horse_names = None, 
                str_aft = None, 
                batch_size = 16, 
                desired_size = 128,
                org_im_width = 672,
                org_im_height = 380):
        self.thresh = thresh

        if self.thresh is not None:
            self.predictor = self.get_predictor()
        
        self.data_path = data_path
        self.out_data_path = out_data_path
        # '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps'
        
        if horse_names is None:
            self.horse_names = ['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']
        else:
            self.horse_names = horse_names
        
        if str_aft is None:
            self.str_aft = '_frame_index.csv'
        else:
            self.str_aft = str_aft

        self.batch_size = batch_size
        self.buffer_size = 0.1
        self.desired_size = desired_size
        self.org_im_height = org_im_height
        self.org_im_width = org_im_width
        # return
        mean_vals = (0.485, 0.456, 0.406)
        self.mean_vals = [int(val*256) for val in mean_vals]

        self.viewpoints_file = '../metadata/viewpoints.csv'
        self.bg_dir = '../data/median_bg_672_380'



    def get_predictor(self):
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.thresh  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        return predictor

    def get_horse_ims(self, horse_name, data_path = None, str_aft = None):
        if data_path is None:
            data_path = self.data_path

        if str_aft is None:
            str_aft = self.str_aft

        csv_path = os.path.join(data_path, horse_name+str_aft)
        frame_df = pd.read_csv(csv_path)
        im_files = []
        for idx_row, row in frame_df.iterrows():
            im_path = util.get_image_name(row['subject'], row['interval_ind'], row['interval'], \
                                         row['view'], row['frame'], data_path)
            if os.path.exists(im_path):
                im_files.append(im_path)
        return im_files

    def save_detections(self):
        print (self.horse_names)
        for horse_name in self.horse_names:
            im_files = self.get_horse_ims( horse_name)
            print ('Processing Horse {}, with {} images'.format(horse_name, len(im_files)))
            self.batch_process(im_files)


    def batch_process(self, input_images):
        batch_size = self.batch_size
        predictor = self.predictor

        input_images_keep = []
        for im_path in input_images:
            out_dir = os.path.split(im_path)[0]+'_dets'
            out_file = os.path.join(out_dir,os.path.split(im_path)[1][:-4]+'.npz')
            if not os.path.exists(out_file):
                input_images_keep.append(im_path)

        print (len(input_images), len(input_images_keep))
        input_images = input_images_keep

        start_idx_all = range(0,len(input_images),batch_size)
        im_paths_batches = []
        for start_idx in start_idx_all:
            end_idx = start_idx+batch_size
            # print (start_idx, end_idx)
            im_paths_batches.append(input_images[start_idx:end_idx])

        with tqdm(total=len(im_paths_batches)) as pbar:
            for idx_im_path, im_paths in enumerate(im_paths_batches):
                pbar.update(1)
                # print ('idx_im_path', idx_im_path)
                inputs_all = []
                # ims_all = []
                for im_path in im_paths:
                    im = cv2.imread(im_path)
                    height, width = im.shape[:2]
                    # ims_all.append(im)
                    image = predictor.transform_gen.get_transform(im).apply_image(im)
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    # print (image.size())
                    inputs = {"image": image, "height": height, "width": width}
                    inputs_all.append(inputs)

                predictions = predictor.model(inputs_all)
                for idx_pred, pred in enumerate(predictions):
                    im_path = im_paths[idx_pred]

                    instances = pred['instances'].to("cpu")
                    pred_boxes = instances.pred_boxes.tensor.detach().numpy()
                    scores = instances.scores.detach().numpy()
                    pred_classes = instances.pred_classes.detach().numpy()
                    out_dir = os.path.split(im_path)[0]+'_dets'
                    util.mkdir(out_dir)
                    out_file = os.path.join(out_dir,os.path.split(im_path)[1][:-4]+'.npz')
                    np.savez(out_file, pred_boxes = pred_boxes, pred_classes = pred_classes, scores = scores)

    # to do. don't process images with dets already
    def save_crop_and_det(self, arg):
        desired_size = self.desired_size
        buffer_size = self.buffer_size
        mean_vals = self.mean_vals
        (im_file, det_file, out_im_file, out_crop_info_file) = arg

        try:
            loaded_data = np.load(det_file)
            pred_classes = loaded_data['pred_classes']
            scores = loaded_data['scores']
            pred_boxes = loaded_data['pred_boxes']

            if len(pred_classes)==0 or (17 not in pred_classes):
                return 2

            im = cv2.imread(im_file)
            org_im_width = im.shape[1]
            org_im_height = im.shape[0]

            # pick the best horse
            idx_sort = np.argsort(scores)[::-1]
            pred_classes = pred_classes[idx_sort]
            scores = scores[idx_sort]
            pred_boxes = pred_boxes[idx_sort,:]
            idx_horse = np.where(pred_classes==17)[0][0]
            pred_box = pred_boxes[idx_horse,:]

            # make pred_box square
            box_size = np.array([pred_box[2]-pred_box[0],pred_box[3]-pred_box[1]])
            if box_size[0]<box_size[1]:
                diff = (box_size[1]-box_size[0])/2
                to_add = np.array([-diff,0,+diff,0])
            else:
                diff = (box_size[0]-box_size[1])/2
                to_add = np.array([0,-diff,0,+diff])
            pred_box = pred_box+to_add
            pred_box = pred_box.astype(int)
            
            # expand with buffer
            to_expand = buffer_size*box_size
            to_expand = np.array([-to_expand[0],-to_expand[1],+to_expand[0],+to_expand[1]]).astype(int)
            pred_box = pred_box+to_expand
            
            # pad image to complete expansion
            left = -1*min(pred_box[0],0)
            top =  -1*min(pred_box[1],0)
            right = max(0,pred_box[2]-(org_im_width-1))
            bottom = max(0,pred_box[3]-(org_im_height-1))
            to_pad = np.array([top,bottom,left,right])

            mean_vals = mean_vals[::-1]
            # print (mean_vals)
            
            # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value = tuple(mean_vals))
            im_size = (im.shape[1],im.shape[0])

            # shift pred_box for padded image
            pred_box = pred_box+np.array([left,top,left,top])

            # double check box in im
            pred_box[0] = max(0,pred_box[0])
            pred_box[1] = max(0,pred_box[1])
            pred_box[2] = min(im_size[0]-1,pred_box[2])
            pred_box[3] = min(im_size[1]-1,pred_box[3])
            
            # crop im
            im_crop = im[pred_box[1]:pred_box[3],pred_box[0]:pred_box[2]]
            
            # resize crop. switch to PIL for better resize
            im_final = np.array(Image.fromarray(im_crop[:,:,::-1]).resize((desired_size,desired_size), resample=PIL.Image.BICUBIC))
            
            cv2.imwrite(out_im_file, im_final[:,:,::-1])    
            np.savez(out_crop_info_file, to_pad = to_pad, pred_box = pred_box)
            return 1
        except:
            return 0

    def save_all_crop_im(self):
        data_path = self.data_path
        out_data_path = self.out_data_path
        
        for horse_name in self.horse_names:
            print (horse_name)
            im_files = self.get_horse_ims(horse_name)

            args = []
            im_files_used = []
            for idx_im_file, im_file in enumerate(im_files):
                det_file = os.path.join(os.path.split(im_file)[0]+'_dets',os.path.split(im_file)[1][:-4]+'.npz')
                
                out_im_file = im_file.replace(data_path, out_data_path)
                out_crop_info_file = os.path.join(os.path.split(im_file)[0].replace(data_path,out_data_path)+'_cropbox',os.path.split(im_file)[1][:-4]+'.npz')
                
                if os.path.exists(out_im_file) and os.path.exists(out_crop_info_file):
                    continue
                
                util.makedirs(os.path.split(out_im_file)[0])
                util.makedirs(os.path.split(out_crop_info_file)[0])
                
                arg_curr = (im_file, det_file, out_im_file, out_crop_info_file)
                args.append(arg_curr)
                im_files_used.append(im_file)
            
            print (len(args),len(im_files))
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            ret_vals = pool.map(self.save_crop_and_det, args)
            pool.close()
            pool.join()

            # ret_vals = []
            # for arg in args:
            #     ret_vals.append(self.save_crop_and_det(arg))
            #     print (ret_vals)
            #     break
            # return
            
            assert len(ret_vals)== len(im_files_used)
            print ('0:',ret_vals.count(0),', 1:',ret_vals.count(1),', 2:',ret_vals.count(2),', total:',len(ret_vals))
            out_file_log = os.path.join(out_data_path, horse_name+'_im_crop_log.npz')
            np.savez(out_file_log, ret_vals = np.array(ret_vals), im_files_used = np.array(im_files_used))

    def save_bg_crop(self, arg):
        desired_size = self.desired_size
        # mean_vals = self.mean_vals
        (im_file, crop_info_file, out_im_file, mean_vals) = arg
        try:
            loaded_data = np.load(crop_info_file)
            
            to_pad = loaded_data['to_pad']
            pred_box = loaded_data['pred_box']

            im = cv2.imread(im_file)
            
            # pad im
            mean_vals = mean_vals[::-1]
            im = cv2.copyMakeBorder(im, to_pad[0], to_pad[1], to_pad[2], to_pad[3], cv2.BORDER_CONSTANT, value = tuple(mean_vals))
            
            # crop im
            im_crop = im[pred_box[1]:pred_box[3],pred_box[0]:pred_box[2]]
            
            # resize crop. switch to PIL for better resize
            im_final = np.array(Image.fromarray(im_crop[:,:,::-1]).resize((desired_size,desired_size), resample=PIL.Image.BICUBIC))
            
            cv2.imwrite(out_im_file, im_final[:,:,::-1])    
            return 1
        except:
            return 0

    def save_all_crop_flow(self, flow_dir, flow_post_pend, out_flow_post_pend, str_aft = None):
        lookup_viewpoint = pd.read_csv(self.viewpoints_file, index_col='subject')

        if str_aft is None:
            str_aft = self.str_aft

        # desired_size = 128
        # mean_vals = (0.485, 0.456, 0.406)
        # mean_vals = [int(val*256) for val in mean_vals]
    
        # data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop'
        # horse_names = ['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']
        # str_aft = '_frame_index.csv'


        for horse_name in self.horse_names:
            print (horse_name)
            im_files = self.get_horse_ims(horse_name, data_path = self.out_data_path, str_aft = str_aft)

            args = []
            im_files_used = []
            for idx_im_file, im_file in enumerate(im_files):
                crop_info_file = os.path.join(os.path.split(im_file)[0]+'_cropbox',os.path.split(im_file)[1][:-4]+'.npz')
                flow_file = os.path.split(im_file)[1].replace('.jpg','.png')
                out_im_file = os.path.join(os.path.split(im_file)[0]+out_flow_post_pend,flow_file)

                view = os.path.split(im_file)[0][-1]
                camera = lookup_viewpoint.at[horse_name, view]
                bg_file = im_file.replace(self.out_data_path, flow_dir)
                bg_file = os.path.join(os.path.split(bg_file)[0]+flow_post_pend,flow_file)
                # bg_file = os.path.join(self.bg_dir, 'median_0.1fps_camera_{}.jpg'.format(camera-1))

                if os.path.exists(out_im_file):
                    continue
                
                util.makedirs(os.path.split(out_im_file)[0])
                arg_curr = (bg_file, crop_info_file, out_im_file, (0,0,0))
                args.append(arg_curr)
                im_files_used.append(im_file)
            
            print (len(args),len(im_files))

            # for arg in args:
            #     print (arg)
            #     ret_val = self.save_bg_crop(arg)
            #     # print (ret_val)
                
            #     break
            # break

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            ret_vals = pool.map(self.save_bg_crop, args)
            pool.close()
            pool.join()

            assert len(ret_vals)== len(im_files_used)
            print ('0:',ret_vals.count(0),', 1:',ret_vals.count(1),', total:',len(ret_vals))
            out_file_log = os.path.join(self.out_data_path, horse_name+'_bg_crop_log.npz')
            np.savez(out_file_log, ret_vals = np.array(ret_vals), im_files_used = np.array(im_files_used))

    def save_all_crop_bg(self, str_aft = None):
        lookup_viewpoint = pd.read_csv(self.viewpoints_file, index_col='subject')

        if str_aft is None:
            str_aft = self.str_aft

        # desired_size = 128
        # mean_vals = (0.485, 0.456, 0.406)
        # mean_vals = [int(val*256) for val in mean_vals]
    
        # data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop'
        # horse_names = ['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']
        # str_aft = '_frame_index.csv'


        for horse_name in self.horse_names:
            im_files = self.get_horse_ims(horse_name, data_path = self.out_data_path, str_aft = str_aft)

            args = []
            im_files_used = []
            for idx_im_file, im_file in enumerate(im_files):
                crop_info_file = os.path.join(os.path.split(im_file)[0]+'_cropbox',os.path.split(im_file)[1][:-4]+'.npz')
                out_im_file = os.path.join(os.path.split(im_file)[0]+'_bg',os.path.split(im_file)[1])

                view = os.path.split(im_file)[0][-1]
                camera = lookup_viewpoint.at[horse_name, view]
                bg_file = os.path.join(self.bg_dir, 'median_0.1fps_camera_{}.jpg'.format(camera-1))

                if os.path.exists(out_im_file):
                    continue
                
                util.makedirs(os.path.split(out_im_file)[0])
                arg_curr = (bg_file, crop_info_file, out_im_file, self.mean_vals)
                args.append(arg_curr)
                im_files_used.append(im_file)
            
            print (len(args),len(im_files))

            # for arg in args:
            #     ret_val = self.save_bg_crop(arg)
            #     print (ret_val)
            #     print (arg)
            #     break
            # break

            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            ret_vals = pool.map(self.save_bg_crop, args)
            pool.close()
            pool.join()

            assert len(ret_vals)== len(im_files_used)
            print ('0:',ret_vals.count(0),', 1:',ret_vals.count(1),', total:',len(ret_vals))
            out_file_log = os.path.join(self.out_data_path, horse_name+'_bg_crop_log.npz')
            np.savez(out_file_log, ret_vals = np.array(ret_vals), im_files_used = np.array(im_files_used))

def script_checking_horse_dets():
    data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps'
    horse_names = ['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']
    str_aft = '_frame_index.csv'

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    coco_class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None)
    # print (coco_class_names.index('horse'))

    for horse_name in horse_names:
        print (horse_name)
        im_files = get_horse_ims(data_path, horse_name, str_aft)
        det_files = [os.path.join(os.path.split(file_curr)[0]+'_dets',os.path.split(file_curr)[1][:-4]+'.npz') for file_curr in im_files]
        det_dict = {'None':[],'horse':[]}
        for det_file in det_files:
            loaded_data = np.load(det_file)
            pred_classes = loaded_data['pred_classes']
            scores = loaded_data['scores']
            if len(pred_classes)==0:
                det_dict['None'].append(0)
            elif 17 in pred_classes:
                idx_horse = np.where(pred_classes==17)[0][0]
                score_horse = scores[idx_horse]
                det_dict['horse'].append(score_horse)
            else:
                idx_max = np.argmax(scores)
                pred_class = pred_classes[idx_max]
                pred_score = scores[idx_max]
                class_str = coco_class_names[pred_class]
                if class_str not in det_dict.keys():
                    det_dict[class_str] = []
                det_dict[class_str].append(pred_score)

        print ('horse percent',len(det_dict['horse'])/len(im_files))


def main():
    # script_to_save_all_crop_bg()
    data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps'
    out_data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    # str_aft = '_thresh_0.70_frame_index.csv'
    str_aft = '_percent_0.01_frame_index.csv'
    # horse_names = ['aslan','brava']
    # horse_names = ['herrera','inkasso']
    # horse_names = ['kastanjett']
    # horse_names = ['naughty_but_nice','sir_holger']
    # horse_names = ['inkasso']
    horse_names = None
    thresh = None

    hd = HorseDetector(data_path, out_data_path, thresh = thresh, horse_names = horse_names, str_aft = str_aft, batch_size = 12)
    # hd.save_detections()
    # hd.save_all_crop_im()

    # copy csv files and potentially reduce them
    # command = ['cp',os.path.join(data_path,'*'+str_aft),out_data_path+'/']
    # print (' '.join(command))

    # hd.save_all_crop_bg(str_aft = '_reduced'+str_aft)
    hd.save_all_crop_flow( data_path, '_opt','_opt', str_aft = '_reduced'+str_aft)

    



if __name__=='__main__':
    main()