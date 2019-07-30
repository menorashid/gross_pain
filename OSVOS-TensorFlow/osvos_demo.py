from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
sys.path.append('../code')
from helpers import util, visualize
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)

# User defined parameters
gpu_id = 0
train_model = True

# seq_name = "car-shadow"
# result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
# seq_name = "ch01_20181209092600"
# seq_name = 'ch05_20181208230458'
seq_name = 'ch06_20181209093714'
result_path = os.path.join('../experiments/osvos_tensorflow', seq_name)
dir_im = os.path.join('../data/tester_kit/naughty_but_nice', seq_name)
dir_anno = os.path.join('../data/tester_kit/naughty_but_nice',seq_name+'_anno')

util.makedirs(result_path)

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
# parent_path = os.path.join('models', 'car-shadow', 'car-shadow.ckpt-500')
logs_path = os.path.join('models', seq_name)
max_training_iters = 500

# Define Dataset
# test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
# test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
# if train_model:
#     train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
#                   os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
#     dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
# else:
#     dataset = Dataset(None, test_imgs, './')

import glob
test_imgs = glob.glob(os.path.join(dir_im,'*.jpg'))
test_imgs.sort()
test_imgs = test_imgs[::20]
test_imgs = test_imgs[:50]
print (test_imgs[:10])
# test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
if train_model:
    # train_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '00000.jpg')+' '+
                  # os.path.join('DAVIS', 'Annotations', '480p', seq_name, '00000.png')]
    train_imgs = [test_imgs[0]+' '+os.path.join(dir_anno, os.path.split(test_imgs[0])[1].replace('.jpg','.png'))]
    print (train_imgs)
    # raw_input()
    dataset = Dataset(train_imgs, test_imgs, '', data_aug=True)
else:
    dataset = Dataset(None, test_imgs, './')




# Train the network
if train_model:
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        osvos.test(dataset, checkpoint_path, result_path)

# Show results
overlay_color = [255, 0, 0]
transparency = 0.6
out_dir = os.path.join('../scratch',seq_name)
util.mkdir(out_dir)
plt.ion()
for idx_img_p, img_p in enumerate(test_imgs):
    plt.figure()
    frame_num = img_p.split('.')[0]
    # img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
    # mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))

    img = np.array(Image.open(img_p))
    mask = np.array(Image.open(os.path.join(result_path, os.path.split(img_p)[1].replace('.jpg','.png'))))

    mask = mask//np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
    plt.imshow(im_over.astype(np.uint8))
    plt.axis('off')
    plt.show()
    # plt.pause(0.01)
    # plt.clf()
    plt.savefig(os.path.join(out_dir, '%09d'%idx_img_p+'.jpg'))
    plt.close()

visualize.writeHTMLForFolder(out_dir)
