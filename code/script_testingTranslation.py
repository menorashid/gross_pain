import os
import numpy as np
from helpers import util
import pandas as pd

def main():
    rot_folder = '../data/rotation_cal_2'
    lookup_viewpoint = pd.read_csv('../metadata/viewpoints.csv')
    lookup_viewpoint = lookup_viewpoint.set_index('subject')
    rots = []
    tvecs = []
    for view in range(4):
        camera = int(lookup_viewpoint.at['cell1', str(view)])
        rots.append(np.load(os.path.join(rot_folder,'extrinsic_rot_'+str(camera)+'.npy')))
        tvecs.append(np.load(os.path.join(rot_folder,'extrinsic_tvec_'+str(camera)+'.npy')))

    rots = np.array(rots)
    tvecs = np.array(tvecs)
    
    print (np.min(tvecs),np.max(tvecs))
    print (tvecs)

    pts = np.mean(tvecs, axis=0)[:,np.newaxis]
    tvecs = tvecs[:,:,np.newaxis]
    # pair = [0,3]
    # for view in pair:
    [view1, view2] = [0,1]
    p1 = np.matmul(rots[view1],pts+tvecs[view1])
    p2 = np.matmul(rots[view2],pts+tvecs[view2])
    r = np.matmul(rots[view2], rots[view1].T)
    t = np.matmul(rots[view2],tvecs[view2]-tvecs[view1])
    p2_1 = np.matmul(r,p1)+t
    print (pts)
    print (p1)
    print (p2)
    print (p2_1)

    print (tvecs/5)
    print (tvecs/10)
    print (tvecs/20)
    # print (pts.shape)


    # print (viewpoints)

if __name__=='__main__':
    main()