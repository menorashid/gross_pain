import os
try:
    from helpers import util 
except:
    import util
import numpy as np
import glob
from datetime import datetime
import multiprocessing
import pandas as pd


def parse_event(args):
    lines, idx_curr = args
    rel_lines = lines[idx_curr:idx_curr+9]
    date_str = ' '.join(rel_lines[1].split()[1:])
    datetime_object = datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')

    assert rel_lines[3].startswith('Major')
    assert rel_lines[4].startswith('Minor')
    assert rel_lines[5].startswith('Local')
    assert rel_lines[6].startswith('Host')
    assert rel_lines[7].startswith('Parameter')
    if not rel_lines[8].startswith('Camera'):
        return None

    vals = []
    for idx in [3,4,8]:
        line_curr = rel_lines[idx]
        val = line_curr[line_curr.index(':')+1:].strip()
        vals.append(val)

    return [datetime_object]+vals
    
def read_motion_file(file_curr):
    lines = util.readLinesFromFile(file_curr)
    lines = [line.strip('\r') for line in lines]
    lines = np.array(lines)
    idx_event = np.where(lines=='----------------------------')[0]
    idx_event = idx_event[::2]
    p = multiprocessing.Pool()
    args = [(lines, idx_curr) for idx_curr in idx_event]
    events = p.map(parse_event, args)
    events = [event for event in events if event is not None]
    columns = np.array(['Date Time','Major Type', 'Minor Type', 'Camera'])
    df = pd.DataFrame(columns = columns)
    for idx in range(len(events)):
        df.loc[idx]=events[idx]
    
    return df
    
def get_vid_info(vid_file):
    cam_time = os.path.split(vid_file)[1]
    if '.' in cam_time:
        cam_time = cam_time[:cam_time.rindex('.')]
    
    cam_time = cam_time.split('_')
    cam = int(cam_time[0][2:])
    vid_start_time = datetime.strptime(cam_time[1],'%Y%m%d%H%M%S')
    vid_start_time = np.datetime64(vid_start_time)
    # print (type(vid_start_time))
    return cam, vid_start_time



def create_video_dframe(horse_dir):
    horse_name = os.path.split(horse_dir)[1]
    
    str_vid_file = 'ch'+'[0-9]'*2+'_'+'[0-9]'*14+'.mp4'
    vid_files = glob.glob(os.path.join(horse_dir,'*',str_vid_file))
    vid_files.sort()
    vid_file = vid_files[0]

    pool = multiprocessing.Pool()
    vid_infos = pool.map(get_vid_info, vid_files)
    
    cam, vid_start_time = get_vid_info(vid_file)

    columns = np.array(['Horse','Camera', 'Start Time'])
    df = pd.DataFrame(columns = columns)
    for idx,vid_info in enumerate(vid_infos):
        df.loc[idx]=[horse_name]+list(vid_info)
    
    return df

    # get_vid_start_time(os.path.split(vid_file)[1])
    # get_vid_start_time(os.path.split(vid_file)[1][:-4])


# def get_closest_vid_time(vid_df, 
#     pass

def main():

    in_dir = '../data/surveillance_camera/Naughty_but_Nice'
    motion_file = glob.glob(os.path.join(in_dir,'*','*.txt'))[700]
    print (motion_file)
    vid_df = create_video_dframe(in_dir)
    
    motion_df = read_motion_file(motion_file)
    motion_str = 'Motion Detection Started'
    rows = motion_df.loc[(motion_df['Minor Type']==motion_str),['Date Time']]
    motion_times =rows.iloc[:,0].values    
    motion_time = motion_times[3]
    print (motion_time, motion_time)

    
    for cam_curr in range(1,9):
        bin_keep = vid_df['Camera']==cam_curr
        if np.sum(bin_keep)==0:
            print ('Camera',cam_curr,'not found')
        else:
            vid_times = vid_df.loc[bin_keep & (vid_df['Start Time']<=motion_time),['Start Time']]
            vid_times = vid_times.iloc[:,0].values
            vid_diffs = motion_time - vid_times
            closest_idx = np.argmin(vid_diffs)
            print ('Camera',cam_curr,'closest_idx')
            print (vid_diffs[closest_idx])
            print (vid_times[closest_idx])

    for cam_curr in range(1,9):
        bin_keep = vid_df['Camera']==cam_curr
        if np.sum(bin_keep)==0:
            print ('Camera',cam_curr,'not found')
        else:
            vid_times = vid_df.loc[bin_keep & (vid_df['Start Time']<=motion_time),['Start Time']]
            vid_times = vid_times.iloc[:,0].values
            vid_diffs = motion_time - vid_times
            closest_idx = np.argmin(vid_diffs)
            print ('Camera',cam_curr,'closest_idx')
            print (vid_diffs[closest_idx])
            print (vid_times[closest_idx])


    # cameras = range(1,9)




    # in_dir = '../data/tester_kit/naughty_but_nice'
    # motion_files = glob.glob(os.path.join(in_dir,'*.txt'))
    # motion_files.sort()
    # motion_file = motion_files[2]
    # print (motion_file)
    # df = read_motion_file(motion_file)
    # print (df[:10])


if __name__=='__main__':
    main()