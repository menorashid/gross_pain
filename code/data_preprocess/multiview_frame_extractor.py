import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import pandas as pd
import numpy as np
import subprocess
import torch
import os
from helpers import util
import multiprocessing
import time as time_proper

VIEWPOINTS = {0: 'Front left', 1: 'Front right',
              2: 'Back right', 3: 'Back left'}
LEN_FILE_ID = 19


class MultiViewFrameExtractor():
    def __init__(self, data_path, width= None, height = None, frame_rate = None, output_dir = None,
                 views = None, data_selection_path = None, num_processes = None, offset_file = None):
        """
        Args:
        data_path: str (where the raw videos are, root where the structure is root/subject/)
        width: int
        height: int
        frame_rate: int (frames per second to extract)
        output_dir: str (root folder where to save all frames)
        views: [int] (which viewpoints to include in the dataset, e.g. all [0,1,2,3])
                       The viewpoints are indexed starting from "front left" (FL=0) and
                       then clock-wise -- FR=1, BR=2, BL=3. Front being the side facing
                       the corridor, and R/L as defined from inside of the box.)
        data_selection_path: str (path to .csv-file containing the data selection)
                             The .csv-file should specify the subject, start and end
                             date-time for the intervals, and a pain label.
        num_processes: int (number of processes to use for frame extraction)
        """
        self.data_path = data_path
        self.image_size = (width, height)
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.views = views
        if data_selection_path is None:
            self.data_selection_df = None
            self.subjects = None
        else:
            self.data_selection_df = pd.read_csv(data_selection_path)
            self.subjects = self.data_selection_df.subject.unique()
        self.num_processes = num_processes
        self.lookup_viewpoint = pd.read_csv('../metadata/viewpoints.csv', index_col='subject')
        if offset_file is None:
            self.offset_df = None
        else:
            self.offset_df = pd.read_csv(offset_file)

    # The following methods construct path strings. Frames saved on format:
    # self.output_dir/subject/yymmddHHMMSS_HHMMSS/LXYZ_%06d.jpg
    # where the two timestamps are for the start and end of the interval,
    # L is the first letter of the subject name,
    # X is the index of the interval for that subject,
    # Y is the viewpoint and
    # Z is the index of the raw video file for that interval.

    def get_subject_dir_path(self, subject):
        return os.path.join(self.output_dir,subject)

    def get_interval_dir_path(self, subject_dir_path, start, end):
        # Keep only HHMMSS for the end time.
        interval_str = start + '_' + end[8:]
        return os.path.join(subject_dir_path, interval_str), interval_str

    def get_view_dir_path(self, interval_dir_path, view):
        return os.path.join(interval_dir_path,str(view))

    def extract_frames(self, replace = True, subjects_to_extract = None):
        ffmpeg_scale = 'scale='+ str(self.image_size[0]) + ':' + str(self.image_size[1])
        inc = pd.Timedelta(1/float(self.frame_rate),'s')
        
        if subjects_to_extract is None:
            subjects_to_extract = self.subjects
        
        # To help us read the data, we save a long .csv-file
        # with labels for each frame(a frame index).
        column_headers = ['interval', 'interval_ind', 'view', 'subject', 'pain', 'frame']
        

        for i, subject in enumerate(subjects_to_extract):
            big_list = []

            subject_dir_path = self.get_subject_dir_path(subject)
            print("Extracting frames for subject {}...".format(subject))
            interval_ind = 0  # Counter for the extracted intervals to index file extension.
            horse_df = self.data_selection_df.loc[self.data_selection_df['subject'] == subject]

            for ind, row in horse_df.iterrows():
                # Each row will contain the start and end times and a pain label.
                print(row)
                pain = row['pain']
                # Just for directory naming
                start_str = str(row['start'])
                end_str = str(row['end'])
                interval_dir_path, interval_str = self.get_interval_dir_path(subject_dir_path,
                                                                             start_str,
                                                                             end_str)

                # Timestamps for start and end
                start_interval = pd.to_datetime(row['start'], format='%Y%m%d%H%M%S')
                end_interval = pd.to_datetime(row['end'], format='%Y%m%d%H%M%S')
                interval_duration = end_interval - start_interval
                print('\n')
                print('Total interval duration: ', interval_duration)

                # set up process pool
                pool = multiprocessing.Pool(self.num_processes)

                # calculate all times for extraction
                all_times = []
                curr_time = start_interval
                while curr_time<end_interval:
                    all_times.append(curr_time)
                    curr_time += inc

                print('Total frames in interval: ', len(all_times))
                
                for view in self.views:
                    t = time_proper.time()
                    print('Doing view', view)

                    view_dir_path = self.get_view_dir_path(interval_dir_path, view)
                    util.makedirs(view_dir_path)

                    # get video_names and times
                    video_paths, video_start_times = self._find_videos(subject, view, start_interval.date())
                    if len(video_paths)==0:
                        print('No videos for subject %s, view %d, date %s found'%(subject, view, str(start_interval.date())))
                        continue

                    # get containing videos multithreaded
                    # to do if need be

                    # get containing videos
                    print('Getting video for each time ')
                    path_and_time = []
                    for curr_time in all_times:
                        video_path, time_in_video = self._find_video_containing_time(video_paths, video_start_times, curr_time)
                        if video_path is None:
                            print ('Video for time ',curr_time,' not found')
                        else:
                            path_and_time.append((video_path, time_in_video))

                    # set up ffmpeg commands
                    ffmpeg_commands = []
                    for idx_path, (video_path, time_in_video) in enumerate(path_and_time):
                        time_pd = pd.Timedelta(time_in_video)
                        
                        # key frames are once every 2.5 seconds. go to the nearest one, and then accurately seek.
                        time_kf = pd.Timedelta(time_pd.total_seconds()//2.5 *2.5,'s')
                        diff = time_pd - time_kf
                        time_in_video = str(time_kf)[7:]
                        # diff_str = str(diff)[7:]
                        diff = diff.total_seconds()*20
                        diff = diff-1 if diff>0 else 0
                        frame_num_str = r'select=eq(n\,%d)'%diff

                        # ffmpeg -ss 00:25:12.5 -i ../data/lps_data/surveillance_camera/brava/2018-11-02/ch04_20181102161155.mp4 -vframes 1 -y -vf "select=eq(n\,29)",scale=2688:1520 ../scratch/frame_extraction_with_offsets/brava/20181102163705_164705/2/br_00_2_000005.jpg -hide_banner
                        
                        padded_idx_path = '%06d'%idx_path
                        padded_interval_ind = '%02d'%interval_ind
                        frame_id = '_'.join([subject[:2], padded_interval_ind, str(view), padded_idx_path])
                        complete_output_path = os.path.join(view_dir_path, frame_id + '.jpg')

                        if not replace and os.path.exists(complete_output_path):
                            continue
                        
                        # Save a row to the frame index .csv-file
                        row_list = [interval_str, padded_interval_ind, view, subject, pain, padded_idx_path]
                        big_list.append(row_list)

                        ffmpeg_command = ['ffmpeg', '-ss', time_in_video, '-i', video_path,
                                          '-vframes','1',
                                          '-y',
                                          '-vf', frame_num_str+','+ffmpeg_scale,
                                          complete_output_path,
                                          '-hide_banner', '-loglevel', 'quiet']

                                            # '-ss',diff_str,
                        # print (' '.join(ffmpeg_command))
                        ffmpeg_commands.append(ffmpeg_command)
                        # break

                    # extract
                    # print (ffmpeg_commands[0])
                    # subprocess.call(ffmpeg_commands[0])
                    # s = input()
                    # break
                    print('Extracting %d frames multithreaded '%len(ffmpeg_commands))
                    pool.map(subprocess.call,ffmpeg_commands)
                    print('Done for view', view, time_proper.time()-t )
                    

            
                
                pool.close()
                pool.join()
                print('Done for interval', interval_ind)
                print('\n')
                interval_ind += 1

            frame_index_df = pd.DataFrame(big_list, columns=column_headers)
            out_file_index = os.path.join(self.output_dir,subject+'_'+'frame_index.csv')
            print(out_file_index)
            frame_index_df.to_csv(path_or_buf= out_file_index)

        # reset because ffmpeg makes the terminal messed up
        # subprocess.call('reset')

    def get_videos_containing_intervals(self, subjects_to_extract = None):
        
        if subjects_to_extract is None:
            subjects_to_extract = self.subjects
        
        video_paths = []

        for i, subject in enumerate(subjects_to_extract):
            horse_df = self.data_selection_df.loc[self.data_selection_df['subject'] == subject]

            for ind, row in horse_df.iterrows():
                print (row)
                # Each row will contain the start and end times and a pain label.
                start_str = str(row['start'])
                end_str = str(row['end'])
                _, interval_str = self.get_interval_dir_path('',start_str,end_str)
                # print (interval_str)

                # Timestamps for start and end
                start_interval = pd.to_datetime(row['start'], format='%Y%m%d%H%M%S')
                end_interval = pd.to_datetime(row['end'], format='%Y%m%d%H%M%S')
                interval_duration = end_interval - start_interval
                print('\n')
                print('Total interval duration: ', interval_duration)

                for view in self.views:
                    curr_time = start_interval
                    while curr_time< end_interval:
                        video_path, remaining_duration, time_in_video = self._find_video_and_its_duration(subject, view, curr_time)
                        curr_time = curr_time+remaining_duration
                        video_paths.append(video_path)

                    video_end, _, _ = self._find_video_and_its_duration(subject, view, end_interval)
                    # print (video_end, video_paths[-1],curr_time,end_interval)
                    if video_end !=video_paths[-1]:
                        video_paths.append(video_end)
                        print ('potential problem', video_paths[-1], video_end, end_interval)
                    # assert (video_end == video_paths[-1])

        return video_paths
                    

                        

    def extract_single_time(self, subject, time_str, views, out_dir):
        time_pd = from_filename_time_to_pd_datetime(time_str)
        out_files = []
        for view in views:
            # get video_names and times
            video_paths, video_start_times = self._find_videos(subject, view, time_pd.date())
            video_path, time_in_video = self._find_video_containing_time(video_paths, video_start_times, time_pd)

            
            frame_id = '_'.join([subject[:2], str(view),time_str+'.jpg'])
            complete_output_path = os.path.join(out_dir, frame_id)
            ffmpeg_command = ['ffmpeg', '-ss', time_in_video, '-i', video_path, 
                                  '-vframes','1',
                                  '-y',
                                  complete_output_path,
                                  ]
                                  
            print (' '.join(ffmpeg_command))
            subprocess.call(ffmpeg_command)
            out_files.append(complete_output_path)

        return out_files


    def create_clip_directories(self):
        # subprocess.call(['mkdir', self.output_dir])
        util.mkdir(self.output_dir)
        for i, subject in enumerate(self.subjects):
            subject_dir_path = self.get_subject_dir_path(subject)
            # subprocess.call(['mkdir', subject_dir_path])
            util.mkdir(subject_dir_path)

            print('\n')
            print("Creating clip directories for subject {}...".format(subject))

            horse_df = self.data_selection_df.loc[self.data_selection_df['subject'] == subject]
            for ind, row in horse_df.iterrows():
                start = str(row['start'])
                end = str(row['end'])
                interval_dir_path, _ = self.get_interval_dir_path(subject_dir_path, start, end)
                # subprocess.call(['mkdir', interval_dir_path])
                util.mkdir(interval_dir_path)
                for view in self.views:
                    view_dir_path = self.get_view_dir_path(interval_dir_path, view)
                    # subprocess.call(['mkdir', view_dir_path])
                    util.mkdir(view_dir_path)


    def _find_videos(self, subject, view, query_date):
        """
        subject: str (e.g. Aslan)
        view: int (e.g. 0, use lookup table)

        returns [video paths], [video start times pd.TimeDelta]
        """

        # The videos are found in the following dir structure:
        # subject/yyyy-mm-dd/ch0x_yyyymmddHHMMSS.mp4
        # where x is the camera ID for that horse, found in ../metadata/viewpoints.csv

        subject_path = os.path.join(self.data_path, subject)
        camera = self.lookup_viewpoint.at[subject, str(view)]
        camera_str = 'ch0' + str(camera)

        date_dir = [dd for dd in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path,dd)) and 
                    pd.to_datetime(dd.split('/')[-1]).date() == query_date]
        date_path = os.path.join(subject_path, date_dir[0])

        # Get list of all filenames on format 'ch0x_yyyymmddHHMMSS.mp4' 
        camera_filenames = [fn for fn in os.listdir(date_path)
                            if fn.startswith(camera_str) and '.mp4' in fn]


        # get video start time from name
        time_stamps = list(map(self._get_video_start_time, camera_filenames))
        # [self._get_video_start_time(camera_filename
        # list(map(from_filename_time_to_pd_datetime, filename_times))
        
        file_paths = [os.path.join(date_path, file_name) for file_name in camera_filenames]
        return file_paths, time_stamps

    def _find_video_containing_time(self, camera_filenames, time_stamps, time):
        

        # Get the pd.TimeDeltas for each timestamp in the list,
        # relative to the given time
        time_deltas = [get_timedelta(time, ts) for ts in time_stamps]

        # Get the index before that of the first negative TimeDelta
        correct_index = self._get_index_for_correct_file(time_deltas)
        
        if correct_index is None:
            # if time not found return None
            return None, None
        else:
            # Also return the start time as a string for ffmpeg
            # (Remove "0 days " in the beginning of the timedelta to fit the ffmpeg command.)
            start_time_in_video = str(time - time_stamps[correct_index])[7:]
            
            return camera_filenames[correct_index], start_time_in_video


    def _find_video_and_its_duration(self, subject, view, time):
        """
        subject: str (e.g. Aslan)
        view: int (e.g. 0, use lookup table)
        time: pd.datetime
        returns video path str, remainder duration pd.TimeDelta,
                start time in video (str)
        """

        camera_filenames, time_stamps = self._find_videos( subject, view, time.date())
        
        # Get the pd.TimeDeltas for each timestamp in the list,
        # relative to the given time
        time_deltas = [get_timedelta(time, ts) for ts in time_stamps]

        # Get the index before that of the first negative TimeDelta
        correct_index = self._get_index_for_correct_file(time_deltas)

        # Return the length of the remainder of the clip
        remaining_duration = time_stamps[correct_index+1] - time

        # Also return the start time as a string for ffmpeg
        # (Remove "0 days " in the beginning of the timedelta to fit the ffmpeg command.)
        start_time_in_video = str(time - time_stamps[correct_index])[7:]
        
        return camera_filenames[correct_index], remaining_duration, start_time_in_video


    def _get_index_for_correct_file(self, time_deltas):
        """
        time_deltas: [pd.TimeDelta] list of timedeltas.
        Finds index of the file we're looking for; aka the one
        before the one where the delta is negative.
        return: int
        """

        # finds index with the smallest positive time delta. 
        # Returns none if there are no positive time deltas
        # works if there are no negative time deltas.

        time_deltas = np.array([time_delta.total_seconds() for time_delta in time_deltas])
        row_bin = time_deltas >=0
        
        if np.sum(row_bin)==0:
            correct_index = None
        else:
            correct_index = np.argmin(time_deltas[row_bin])
            correct_index = np.where(row_bin)[0][correct_index]

        return correct_index

    def _get_video_start_time(self,vid_name):
        # Remove .mp4 extension, while avoiding some files that are
        # ch0x_yyyymmddHHMMSS_001.mp4.
        # Get only the time part, format now 'yyyymmddHHMMSS'
        # Start from after ch0x_ (index 5)
        vid_name =os.path.split(vid_name)[1]
        vid_name = vid_name[:vid_name.rindex('.')]
        # assert vid_name.count('_')==1
        vid_time = from_filename_time_to_pd_datetime(vid_name[5:LEN_FILE_ID])
        
        # # Convert strings to pd.TimeStamps
        # time_stamps = list(map(from_filename_time_to_pd_datetime, filename_times))

        if self.offset_df is not None:
            rows = self.offset_df.loc[self.offset_df['video_name'] == vid_name]
            if len(rows)!=0:
                assert len(rows)==1
                offset = rows.iloc[0]['offset']
                if offset>=0:
                    vid_time = vid_time - pd.to_timedelta(offset, unit='s')
                else:
                    # print (vid_name, vid_time, offset)
                    vid_time = vid_time +  pd.to_timedelta(-1*offset, unit='s')
                    # print (vid_name, vid_time, offset)
                    # s = input()
            # else:
            #     print ('PROBLEM! video offset not found!')

        return vid_time



def get_timedelta(time_stamp_1, time_stamp_2):
    return time_stamp_1 - time_stamp_2


def from_filename_time_to_pd_datetime(yyyymmddHHMMSS):
    return pd.to_datetime(yyyymmddHHMMSS, format='%Y%m%d%H%M%S')


def main():

    data_path = '../data/lps_data/surveillance_camera'
    
    # data_selection_path = '../metadata/intervals_for_empty.csv'
    data_selection_path = '../data/frame_extraction_files/pain_no_pain_x5h_intervals_for_extraction.csv'
    
    out_dir_offsets = '../metadata/fixing_offsets_with_cam_on'
    out_file_final =  os.path.join(out_dir_offsets,'video_offsets_final.csv')
    
    fps = 0.01
    width = 672
    height = 380
    fname = os.path.split(data_selection_path)[1][:-4]
    fname = '_'.join([str(val) for val in [fname,width,height,fps,'fps']])
    out_dir_testing = os.path.join('../data',fname)
    print (out_dir_testing)
    mve = MultiViewFrameExtractor(data_path = data_path, width= width, height = height, frame_rate = fps, output_dir = out_dir_testing,views = [0,1,2,3], data_selection_path = data_selection_path, num_processes = multiprocessing.cpu_count(), offset_file = out_file_final)
    
    mve.extract_frames(replace = False)

if __name__=='__main__':
    main()