import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
from helpers import util,visualize
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd    
import PIL
from PIL import Image

def get_angle(vec1,vec2):
    dot = np.dot(vec1.T,vec2)
    norms = np.linalg.norm(vec1)* np.linalg.norm(vec2)
    dot = dot/norms
    angle = np.arccos(dot)
    return angle

def script_get_rot():
    meta_dir = '../data_other/camera_calibration_frames_try2'
    intrinsics_dir = os.path.join(meta_dir,'intrinsics')
    npz_name = '1_0.npz'
    im_dir = os.path.join(meta_dir, 'cell1/20200428104445_113700/0')
    file_curr = os.path.join(intrinsics_dir,npz_name)

    intrinsics = np.load(file_curr)
    k = intrinsics['mtx']
    # print (k)

    im_file = os.path.join(im_dir,'ce_00_0_000000.jpg')
    im = cv2.imread(im_file,cv2.IMREAD_GRAYSCALE)
    print (im.shape)
    
    old_center = k[:2,2]    
    print ('old_center',old_center)

    new_center = old_center+np.array([500,700])
    print ('new_center',new_center)

    fx =  k[0,0]
    fy = k[1,1]
    # print ('fx,fy',fx,fy)
    new_x = (new_center[0]-old_center[0])/fx
    new_y = (new_center[1] - old_center[1])/fy

    o = np.array([0,0,1])
    xy = np.array([new_x,new_y,1])

    axis = np.cross(o,xy)
    angle = get_angle(o,xy)
    print (angle)
    print (axis,np.linalg.norm(axis))
    axis = axis/np.linalg.norm(axis)
    print (axis)
    axis = axis*angle
    print (np.linalg.norm(axis))
    r = R.from_rotvec(axis)
    ans = r.apply(o)
    print (ans)
    print (ans/ans[2])
    print (xy)
    im_check = np.matrix(k)*ans[:,np.newaxis]
    im_check = im_check/im_check[2]
    print (new_center)
    print (im_check)
    
def plot_rectangle(im, pred_box):
    im = cv2.rectangle(im, (pred_box[0],pred_box[1]), (pred_box[2],pred_box[3]), (255,255,255),5)
    return im

def get_rot(cell, view, new_center):
    meta_dir = '../data_other/camera_calibration_frames_try2'
    intrinsics_dir = os.path.join(meta_dir,'intrinsics')
    npz_name = cell+'_'+view+'.npz'
    file_curr = os.path.join(intrinsics_dir, npz_name)

    intrinsics = np.load(file_curr)
    k = intrinsics['mtx']
    # k[2,2] = 1
    old_center = k[:2,2]  
    new_center = np.array(new_center)

    # print ('old_center',old_center)
    # print ('new_center',new_center)

    fx =  k[0,0]
    fy = k[1,1]
    
    new_x = (new_center[0]-old_center[0])/fx
    new_y = (new_center[1] - old_center[1])/fy

    o = np.array([0,0,1])
    xy = np.array([new_x,new_y,1])

    axis = np.cross(o,xy)
    angle = get_angle(o,xy)
    
    axis = axis/np.linalg.norm(axis)
    axis = axis*angle
    r = R.from_rotvec(axis)
    ans = r.apply(o)
    # print (ans)
    # print (ans/ans[2])
    # print (xy)
    im_check = np.matrix(k)*ans[:,np.newaxis]
    im_check = im_check/im_check[2]
    # print (new_center)
    # print (im_check)
    rot = r.as_matrix()
    return rot, k

def get_original_pred_box(det_file):
    loaded_data = np.load(det_file)
    pred_classes = loaded_data['pred_classes']
    scores = loaded_data['scores']
    pred_boxes = loaded_data['pred_boxes']

    if len(pred_classes)==0 or (17 not in pred_classes):
        return None

    # pick the best horse
    idx_sort = np.argsort(scores)[::-1]
    pred_classes = pred_classes[idx_sort]
    scores = scores[idx_sort]
    pred_boxes = pred_boxes[idx_sort,:]
    idx_horse = np.where(pred_classes==17)[0][0]
    pred_box = pred_boxes[idx_horse,:]
    return pred_box

def warp_box(pred_box, homo):
    [x_min,y_min,x_max,y_max] = pred_box

    pred_box = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]).T
    pred_box_in = np.array(pred_box)
    pred_box = np.concatenate([pred_box,np.ones((1,pred_box.shape[1]))],axis = 0)
    pred_box_copy = np.array(pred_box)
    pred_box = np.array(np.matrix(homo)*pred_box)
    pred_box = pred_box/pred_box[2:,:]
    pred_box = pred_box[:2,:]
    pred_box = np.array([np.min(pred_box,axis = 1),np.max(pred_box,axis = 1)]).ravel()
    return pred_box

def get_pad_box(pred_box, buffer_size, org_im_width, org_im_height):
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
    return pred_box, to_pad


def main():
    # get_rot()
    # (1520, 2688)
    # get rel cam matrix from horse and view
    
    # figure out new rotated corners of bbox 
    # use to create homography
    # apply homography

    out_dir = '../scratch/checking_crop_shit'
    util.mkdir(out_dir)

    meta_dir = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/aslan/20190103122542_130850'
    meta_dir_im = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps/aslan/20190103122542_130850'
    view = '1'
    
    cell = '1' #to do convert to cam num

    im_name = 'as_00_1_000006'
    desired_size = 128
    buffer_size = 0.1

    im_file = os.path.join(meta_dir_im,view,im_name+'.jpg')
    det_file = os.path.join(meta_dir_im,view+'_dets',im_name+'.npz')


    mean_vals = (0.485, 0.456, 0.406)
    mean_vals = [int(val*256) for val in mean_vals]
    
    # crop_file = os.path.join(meta_dir,view+'_cropbox',im_name+'.npz')
    # loaded_data = np.load(crop_file)
    # to_pad = loaded_data['to_pad']
    # pred_box = loaded_data['pred_box']

    im = cv2.imread(im_file)
    pred_box = get_original_pred_box(det_file)    
    pred_box = pred_box.astype(int)
    center = ((pred_box[2] + pred_box[0])//2, (pred_box[3] + pred_box[1])//2)

    im = cv2.circle(im,center,5,(255,255,255),-1)
    cv2.imwrite(os.path.join(out_dir,'im.jpg'),im)

    center_big = np.array(center)*4
    
    rot, k = get_rot(cell, view, center_big)
    k[2,2] = 4  
    
    k_inv = np.linalg.inv(k)
    homo = np.matrix(k)*np.matrix(rot)*np.matrix(k_inv)
    homo = np.linalg.inv(homo)
    homo = np.array(homo)
    
    
    im = cv2.rectangle(im, (pred_box[0],pred_box[1]), (pred_box[2],pred_box[3]), (255,0,0),5)
    cv2.imwrite(os.path.join(out_dir,'im_rect.jpg'),im)    

    # apply homography on it
    # change warp box to rectangle
    pred_box_warp = warp_box(pred_box, homo)
    pred_box_warp = pred_box_warp.astype(int)

    # apply homography on image
    mean_vals = mean_vals[::-1]
    im_warp = cv2.warpPerspective(im, homo, (im.shape[1], im.shape[0]),cv2.BORDER_CONSTANT, borderValue = tuple(mean_vals))
    im_warp = cv2.rectangle(im_warp, (pred_box_warp[0],pred_box_warp[1]), (pred_box_warp[2],pred_box_warp[3]), (0,0,255),5)
    cv2.imwrite(os.path.join(out_dir,'im_warp_rect.jpg'),im_warp)    
    
    # do the old padding etc
    # make pred_box square
    pred_box, to_pad = get_pad_box(pred_box_warp, buffer_size, im.shape[1], im.shape[0])  

    # mean_vals = mean_vals[::-1]
    # print (mean_vals)

    [top, bottom, left, right] = to_pad
    # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)
    im = cv2.copyMakeBorder(im_warp, top, bottom, left, right, cv2.BORDER_CONSTANT, value = tuple(mean_vals))
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
    
    cv2.imwrite(os.path.join(out_dir,'final_crop.jpg'), im_final[:,:,::-1])    

    # save the new image

    # save pred box and to pad as before
    # save homo and rot





    # crop im
    # im_crop = im[pred_box[1]:pred_box[3],pred_box[0]:pred_box[2]]



    # im_padded = 
    # print (im.shape)
    visualize.writeHTMLForFolder(out_dir)










if __name__=='__main__':
    main()