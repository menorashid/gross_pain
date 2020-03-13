import sys
sys.path.append('.')
import os
import numpy as np
from helpers import util
import xml.etree.ElementTree as et

def main():
    
    w0 = util.readLinesFromFile('figuring_h36/w0.txt')
    w0 = w0[0]
    w0 = w0.replace('[','').replace(']','')
    w0 = [float(val) for val in w0.split()]
    print (len(w0))

    c = 1
    s = 1
    w1 = np.zeros((15,));

    params = np.zeros((4,11,15))

    for c in range(1,5):
        for s in range(1,12):
            start = 6*((c-1)*11 + (s-1))
            end = start+6
            os = 264+(c-1)*9
            oe = 264+c*9
            params[c-1, s-1, 0:6 ] = w0[start:end]
            params[c-1, s-1, 6: ] = w0[os:oe]


    print (params[:,1,6:7])
    l = input()
    rot_params = params[:,:5,:3]
    rot_params = np.reshape(rot_params,(rot_params.shape[0]*rot_params.shape[1],3))
    
    for idx in range(rot_params.shape[0]):
        rot_params_curr = rot_params[idx,:]*180/np.pi
        theta = rot_params[idx,0]
        phi = rot_params[idx,1]
        psi = rot_params[idx,2]
        print (rot_params_curr)

        # Ax = np.matrix([[1, 0, 0],
        #                     [0, np.cos(theta), -np.sin(theta)],
        #                     [0, np.sin(theta), np.cos(theta)]])
        # Ay = np.matrix([[np.cos(phi), 0, np.sin(phi)],
        #                 [0, 1, 0],
        #                 [-np.sin(phi), 0, np.cos(phi)]])
        # Az = np.matrix([[np.cos(psi), -np.sin(psi), 0],
        #                 [np.sin(psi), np.cos(psi), 0],
        #                 [0, 0, 1], ])
        # print (Az * Ay * Ax)

        # l = input()
        
        # rot_x = np.array([[1,0,0],
        #        [0, np.cos(rot_params_curr[0]), -np.sin(rot_params_curr[0])],
        #        [0, np.sin(rot_params_curr[0]), np.cos(rot_params_curr[0])]]) 
        # # * 
        # rot_y = np.array([[np.cos(rot_params_curr[1]), 0, np.sin(rot_params_curr[1])],
        #  [0,1,0],
        #  [-np.sin(rot_params_curr[1]), 0, np.cos(rot_params_curr[1])]]) 

        # rot_z = np.array([[np.cos(rot_params_curr[2]), -np.sin(rot_params_curr[2]), 0],
        #  [np.sin(rot_params_curr[2]), np.cos(rot_params_curr[2]), 0],
        #  [0,0,1]])
        # print (np.matmul(np.matmul(rot_x,rot_y),rot_z))



    # start = 6*((c-1)*11 + (s-1))
    # print (start)
    # w1[:6] = w0[start:start+6]
    # print (w1)
    # w1[6:] = w0[(264+(c-1)*9):(264+c*9)];
    # print (w1)

    # print (db)
    # print(db.getroot())
    # print (db.getroot().)
    # obj.frames = xmlobj.frames;
 #      obj.mapping = xmlobj.mapping;
 #      obj.database.cameras = xmlobj.dbcameras;
 #      obj.toffiles = xmlobj.toffiles;
 #      obj.w0, = xmlobj.w0;
 #      obj.skel_angles = xmlobj.skel_angles;
    #       obj.skel_angles.tree = xmlobj.skel_angles.tree'; % for some reason the xml writer gets things transposed
    #       obj.skel_pos = xmlobj.skel_pos;
    #       obj.actionnames = xmlobj.actionnames;
    #       obj.subject_measurements = xmlobj.subject_measurements;
            
 #      for i = 1 : 11
 #        obj.subjects{i} = [];
 #      end
      
 #      obj.actions = 1:16;
 #      obj.cameras = cell(11,4);
 #      obj.subactions = 1:2;
      
 #      obj.train_subjects = xmlobj.train_subjects;
 #      obj.val_subjects = xmlobj.val_subjects;
 #      obj.test_subjects = xmlobj.test_subjects;


if __name__=='__main__':
    main()
