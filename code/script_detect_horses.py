import os
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



def get_predictor(thresh = 0.3):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor



def batch_process(predictor, input_images, batch_size):

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


def get_horse_ims(data_path, horse_name, str_aft):
    csv_path = os.path.join(data_path, horse_name+str_aft)
    frame_df = pd.read_csv(csv_path)
    im_files = []
    for idx_row, row in frame_df.iterrows():
        im_path = util.get_image_name(row['subject'], row['interval_ind'], row['interval'], \
                                     row['view'], row['frame'], data_path)
        if os.path.exists(im_path):
            im_files.append(im_path)
    return im_files



def main():
    data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps'
    horse_names = [['aslan','brava','herrera','inkasso','julia','kastanjett','naughty_but_nice','sir_holger']
    str_aft = '_frame_index.csv'

    predictor = get_predictor()
    for horse_name in horse_names:
        im_files = get_horse_ims(data_path, horse_name, str_aft)
        print (horse_name, len(im_files))
        batch_process(predictor, im_files, 16)
    



if __name__=='__main__':
    main()