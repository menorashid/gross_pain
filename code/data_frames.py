import os
import numpy as np
import glob
from datetime import datetime
import multiprocessing
import pandas as pd
from helpers import util

def get_command_extract_frame_at_time(video_file, out_file, time_curr):
    time_curr = time_curr.astype('datetime64[D]')
    timestampStr = pd.to_datetime(time_curr).strftime('%H:%M:%S')
    command = ['ffmpeg', '-ss']
    command.append(timestampStr)
    command.extend(['-i',video_file])
    command.extend(['-vframes 1'])
    command.append('-vf scale=448:256')
    command.append('-y')
    command.append(out_file)
    command.append('-hide_banner')

    command = ' '.join(command)
    return command

class Horse_Video_Data():
    def __init__(self,horse_data_dir):
        self.data_dir = horse_data_dir
        self.cam_range = range(1,9)
        self.df = self.create_video_dframe()
        

    def create_video_dframe(self):
        horse_name = os.path.split(self.data_dir)[1]
        
        str_vid_file = 'ch'+'[0-9]'*2+'_'+'[0-9]'*14+'.mp4'
        vid_files = glob.glob(os.path.join(self.data_dir,'*',str_vid_file))
        vid_files.sort()

        pool = multiprocessing.Pool()
        vid_infos = pool.map(self.get_vid_info,vid_files)
        
        columns = np.array(['Horse','Camera', 'Start Time','File'])
        df = pd.DataFrame(columns = columns)
        for idx,vid_file in enumerate(vid_files):
            cam, vid_start_time = vid_infos[idx]
            df.loc[idx]=[horse_name, cam, vid_start_time, vid_file]

        return df

    def get_vid_info( self, vid_file):
        cam_time = os.path.split(vid_file)[1]
        if '.' in cam_time:
            cam_time = cam_time[:cam_time.rindex('.')]
        cam_time = cam_time.split('_')
        cam = int(cam_time[0][2:])
        vid_start_time = datetime.strptime(cam_time[1],'%Y%m%d%H%M%S')
        # vid_start_time = np.datetime64(vid_start_time)
        
        return cam, vid_start_time


    def get_closest_vid_time(self, motion_time, cam_range=None, ret_info = ['Camera','Start Time','File']):

        if cam_range is None:
            cam_range = self.cam_range

        assert np.all(np.in1d(ret_info, self.df.columns))

        idx_keep = []

        for cam_curr in cam_range:

            row_bin = self.df['Camera']==cam_curr
            if np.sum(row_bin)==0:
                continue

            row_bin = row_bin & (self.df['Start Time']<=motion_time)

            rows = self.df.loc[row_bin,:]
            vid_times = rows.iloc[:,2].values
            vid_diffs = motion_time - vid_times
            closest_idx = np.argmin(vid_diffs)

            idx_keep_curr = np.where(row_bin)[0][closest_idx]
            idx_keep.append(idx_keep_curr)
            
        rows_to_ret = np.array(self.df.loc[idx_keep,ret_info])
        
        return rows_to_ret



class Motion_File_Data():
    def __init__(self, motion_file):
        self.motion_file = motion_file
        self.df = self.read_motion_file()

    def read_motion_file(self):
        lines = util.readLinesFromFile(self.motion_file)
        lines = [line.strip('\r') for line in lines]
        lines = np.array(lines)
        idx_event = np.where(lines=='----------------------------')[0]
        idx_event = idx_event[::2]
        p = multiprocessing.Pool()
        args = [(lines, idx_curr) for idx_curr in idx_event]
        events = p.map(self.parse_event, args)
        p.close()
        p.join()

        events = [event for event in events if event is not None]
        columns = np.array(['Date Time','Major Type', 'Minor Type', 'Camera'])
        df = pd.DataFrame(columns = columns)
        for idx in range(len(events)):
            df.loc[idx]=events[idx]
        return df

    def parse_event(self, args):
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
    
    def get_motion_times(self,motion_str):
        motion_str = 'Motion Detection Started'
        rows = self.df.loc[(self.df['Minor Type']==motion_str),['Date Time']]
        motion_times =np.array(rows.iloc[:,0])

        return motion_times
    

def main():

    in_dir = '../data/lps_data/surveillance_camera/Naughty_but_Nice'
    
    motion_file = glob.glob(os.path.join(in_dir,'*','*.txt'))[7] 
    
    motion_data = Motion_File_Data(motion_file)
    motion_times = motion_data.get_motion_times('Motion Detection Started')
   
    motion_time = motion_times[-10] 
    
    vid_data = Horse_Video_Data(in_dir)
    closest_times = vid_data.get_closest_vid_time(motion_time)
    out_dir = '../scratch/coordinated_motion'
    util.mkdir(out_dir)

    import subprocess
    from helpers import visualize
    for idx_idx_row,idx_row in enumerate(range(closest_times.shape[0])):
        diff = motion_time - np.datetime64(closest_times[idx_row,1])
        vid_file = closest_times[idx_row,2]

        out_file = os.path.join(out_dir,'cam_'+str(closest_times[idx_row,0])+'.jpg')
        command = get_command_extract_frame_at_time(vid_file, out_file, diff)
        print(command)
        subprocess.call(command, shell=True)

    visualize.writeHTMLForFolder(out_dir)






    


if __name__=='__main__':
    main()

