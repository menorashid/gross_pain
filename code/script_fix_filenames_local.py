import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
from helpers import util

# wrote this because mac book doesn't have a lot of dependancies in other file and has differnt python version. for manual checking of offsets

def record_offsets(args):
    plt.ion()
    plt.figure()
    
    for idx_im,(im_file, out_file) in enumerate(args):
        print idx_im,'of',len(args)
        im = Image.open(im_file).crop((0,0,1000,200))
        im_time = os.path.split(im_file)[1][-10:-4]
        plt.imshow(im)
        title = 'If frame more than vid than NEGATIVE\n'+im_time
        plt.title(title)
        plt.show()

        to_write = []
        s = ''
        while True:
            s = raw_input()
            if s=='q':
                break
            to_write.append(s)

        print out_file,to_write
        util.writeFile(out_file, to_write)

    raw_input()
    plt.close()
    
def fix_offset_files(offset_files):
    problem_files = []
    for offset_file in offset_files:
        lines = util.readLinesFromFile(offset_file)
        assert len(lines)>0
        if len(lines)==1:
            try:
                num = int(lines[0])
                continue
            except:
                problem_files.append(offset_file)
        else:
            problem_files.append(offset_file)

    for problem_file in problem_files:
        print problem_file
        lines = util.readLinesFromFile(problem_file)
        print 'PROBLEM'
        print lines

        to_write = []
        s = ''
        while True:
            s = raw_input()
            if s=='q':
                break
            to_write.append(s)
        print problem_file, to_write

        util.writeFile(problem_file, to_write)


def main():

    out_dir = '../scratch/check_first_frames_with_cam_on'
    out_dir_txt = '../scratch/check_first_frames_with_cam_on_txt'
    out_file_offsets = '../metadata/fixing_offsets_with_cam_on/video_offsets_manual.csv'
    util.mkdir(out_dir_txt)

    times_file = os.path.join(out_dir,'times.npy')
    im_files_to_check = util.readLinesFromFile(os.path.join(out_dir, 'manual_check.txt'))
    args =[]
    out_files = []
    for im_file in im_files_to_check:
        out_file = os.path.join(out_dir_txt, os.path.split(im_file)[1].replace('.jpg','.txt'))
        out_files.append(out_file)
        if not os.path.exists(out_file):
            args.append((im_file, out_file))

    print len(args),len(im_files_to_check)
    # record_offsets(args)
    # fix_offset_files(out_files)

    lines = ['im_file,offset']
    for im_file, offset_file in zip(im_files_to_check, out_files):
        offset = util.readLinesFromFile(offset_file)[0]
        
        # sanity check
        num = int(offset)

        lines.append(im_file+','+offset)
    print lines
    print len(lines)
    print out_file_offsets
    util.writeFile(out_file_offsets, lines)





if __name__=='__main__':
    main()