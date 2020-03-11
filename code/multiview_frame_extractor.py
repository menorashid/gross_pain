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
    def __init__(self, data_path, width, height, frame_rate, output_dir,
                 views, data_selection_path, num_processes):
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
        self.data_selection_df = pd.read_csv(data_selection_path)
        self.subjects = self.data_selection_df.subject.unique()
        self.num_processes = num_processes

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
        return os.path.join(subject_dir_path,start + '_' + end[8:])

    def get_view_dir_path(self, interval_dir_path, view):
        return os.path.join(interval_dir_path,str(view))

    def extract_frames(self):
        ffmpeg_scale = 'scale='+ str(self.image_size[0]) + ':' + str(self.image_size[1])
        inc = pd.Timedelta(1/float(self.frame_rate),'s')

        for i, subject in enumerate(self.subjects):
            subject_dir_path = self.get_subject_dir_path(subject)
            print("Extracting frames for subject {}...".format(subject))
            interval_ind = 0  # Counter for the extracted intervals to index file extension.
            horse_df = self.data_selection_df.loc[self.data_selection_df['subject'] == subject]

            for ind, row in horse_df.iterrows():
                # Each row will contain the start and end times and a pain label.
                print(row)
                # Just for directory naming
                start_str = str(row['start'])
                end_str = str(row['end'])
                interval_dir_path = self.get_interval_dir_path(subject_dir_path,
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
                    
                    view_dir_path = self.get_view_dir_path(interval_dir_path, view)

                    # get video_names and times
                    video_paths, video_start_times = self._find_videos(subject, view, start_interval.date())

                    # get containing videos multithreaded
                    # to do if need be

                    # get containing videos
                    print('Getting video for each time ')
                    path_and_time = []
                    for curr_time in all_times:
                        video_path, time_in_video = self._find_video_containing_time(video_paths, video_start_times, curr_time)
                        path_and_time.append((video_path,time_in_video))
                    
                    # set up ffmpeg commands
                    ffmpeg_commands = []
                    for idx_path,(video_path, time_in_video) in enumerate(path_and_time):
                        frame_id = '_'.join([subject[:2], '%02d'%interval_ind, str(view), '%06d.jpg'%idx_path])
                        complete_output_path = os.path.join(view_dir_path, frame_id)
                        ffmpeg_command = ['ffmpeg', '-ss', time_in_video, '-i', video_path,
                                          '-vframes','1',
                                          '-y',
                                          '-vf', ffmpeg_scale,
                                          complete_output_path,
                                          '-hide_banner',
                                          '-loglevel', 'quiet']
                        ffmpeg_commands.append(ffmpeg_command)

                    
                    # extract
                    print('Extracting frames multithreaded ')
                    pool.map(subprocess.call,ffmpeg_commands)
                    print('Done for view', view)

            
                
                pool.close()
                pool.join()
                print('Done for interval', interval_ind)
                print('\n')
                interval_ind += 1

        # reset because ffmpeg makes the terminal messed up
        # subprocess.call('reset')



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
                interval_dir_path = self.get_interval_dir_path(subject_dir_path, start, end)
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
        lookup_viewpoint = pd.read_csv('../metadata/viewpoints.csv', index_col='subject')
        camera = lookup_viewpoint.at[subject, str(view)]
        camera_str = 'ch0' + str(camera)

        date_dir = [dd for dd in os.listdir(subject_path)
                    if pd.to_datetime(dd.split('/')[-1]).date() == query_date]
        date_path = os.path.join(subject_path, date_dir[0])
       
        # Get list of all filenames on format 'ch0x_yyyymmddHHMMSS.mp4' 
        camera_filenames = [fn for fn in os.listdir(date_path)
                            if fn.startswith(camera_str) and '.mp4' in fn]

        # Remove .mp4 extension, while avoiding some files that are
        # ch0x_yyyymmddHHMMSS_001.mp4.
        # Get only the time part, format now 'yyyymmddHHMMSS'
        # Start from after ch0x_ (index 5)
        filename_times = [fn[5:LEN_FILE_ID] for fn in camera_filenames]

        # Convert strings to pd.TimeStamps
        time_stamps = list(map(from_filename_time_to_pd_datetime, filename_times))

        file_paths = [os.path.join(date_path, file_name) for file_name in camera_filenames]
        return file_paths, time_stamps

    def _find_video_containing_time(self, camera_filenames, time_stamps, time):
        

        # Get the pd.TimeDeltas for each timestamp in the list,
        # relative to the given time
        time_deltas = [get_timedelta(time, ts) for ts in time_stamps]

        # Get the index before that of the first negative TimeDelta
        correct_index = self._get_index_for_correct_file(time_deltas)

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
        for ind, td in enumerate(time_deltas):
            # When we have passed the correct file
            if td.days < 0:
                # Get the index from the last file
                correct_index = ind-1
                break
        return correct_index


def get_timedelta(time_stamp_1, time_stamp_2):
    return time_stamp_1 - time_stamp_2


def from_filename_time_to_pd_datetime(yyyymmddHHMMSS):
    return pd.to_datetime(yyyymmddHHMMSS, format='%Y%m%d%H%M%S')

