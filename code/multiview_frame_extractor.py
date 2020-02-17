import pandas as pd
import numpy as np
import subprocess
import torch
import os


VIEWPOINTS = {0: 'Front left', 1: 'Front right',
              2: 'Back right', 3: 'Back left'}
LEN_FILE_ID = 19


class MultiViewFrameExtractor():
    def __init__(self, data_path, width, height, frame_rate, output_dir,
                 views, data_selection_path):
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
        """
        self.data_path = data_path
        self.image_size = (width, height)
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.views = views
        self.data_selection_df = pd.read_csv(data_selection_path)

        self.subjects = self.data_selection_df.Subject.unique()

    # The following methods construct path strings. Frames saved on format:
    # self.output_dir/subject/yymmddHHMMSS_HHMMSS/LXYZ_%06d.jpg
    # where the two timestamps are for the start and end of the interval,
    # L is the first letter of the subject name,
    # X is the index of the interval for that subject,
    # Y is the viewpoint and
    # Z is the index of the raw video file for that interval.

    def get_subject_dir_path(self, subject):
        return self.output_dir + '/' + subject + '/'

    def get_interval_dir_path(self, subject_dir_path, start, end):
        # Keep only HHMMSS for the end time.
        return subject_dir_path + start + '_' + end[8:] + '/'

    def get_view_dir_path(self, interval_dir_path, view):
        return interval_dir_path + str(view) + '/'

    def extract_frames(self):
        for i, subject in enumerate(self.subjects):
            subject_dir_path = self.get_subject_dir_path(subject)
            print("Extracting frames for subject {}...".format(subject))
            interval_ind = 0  # Counter for the extracted intervals to index file extension.
            horse_df = self.data_selection_df.loc[self.data_selection_df['Subject'] == subject]
    
            for ind, row in horse_df.iterrows():
                # Each row will contain the start and end times and a pain label.
                print(row)
                # Just for directory naming
                start_str = str(row['Start'])
                end_str = str(row['End'])
                interval_dir_path = self.get_interval_dir_path(subject_dir_path,
                                                               start_str,
                                                               end_str)

                # Timestamps for start and end
                start_interval = pd.to_datetime(row['Start'], format='%Y%m%d%H%M%S')
                end_interval = pd.to_datetime(row['End'], format='%Y%m%d%H%M%S')
                interval_duration = end_interval - start_interval
                print('\n')
                print('Total interval duration: ', interval_duration)

                for view in self.views:
                    print('\n')
                    print('View: ', view, ' ({})'.format(VIEWPOINTS[view]))
                    # Path to the directory of this view, which is a subdir to the interval
                    view_dir_path = self.get_view_dir_path(interval_dir_path, view)
                    # Reset the start variable to start_interval for each view.
                    start = start_interval

                    # Variable to keep track of how much of the interval we have covered
                    # since we might traverse multiple clips.
                    duration_cumulative = pd.Timedelta(pd.offsets.Second(0))
                    # Include clip ind in the frame_id so that ffmpeg does not overwrite 
                    # frames when extracting from multiple clips into the same folder.
                    clip_ind = 0

                    while duration_cumulative < interval_duration:
                        print('There are still frames left to extract on the interval.', '\n')

                        # Identify the video containing the start
                        start_video_path, \
                        remaining_duration, \
                        start_in_video = self._find_video_and_its_duration(subject, view, start)
                        print('\n')
                        if remaining_duration > (end_interval-start): # If the end is also in this clip
                            print('The end of the interval is in this video file.')
                            remaining_duration = end_interval-start # Only extract until end of interval
                        print('\n')
                        print('Path to clip: ', start_video_path)
                        print('Duration until end or end of clip: ', remaining_duration)
                        print('Start time in video: ', start_in_video)
                        print('\n')
                        # Extract until the end of this clip
                        # (Remove "0 days " in the beginning of the timedelta to fit the ffmpeg command.)
                        duration_ffmpeg = str(remaining_duration)[7:]
                        print(start, end_interval, duration_ffmpeg)
                        frame_id = subject[:2] + '_' + str(interval_ind) + '_' + str(view) + '_'  + str(clip_ind)
                        # complete_output_path = view_dir_path + '%~nf' + frame_id + '_%06d.jpg'
                        complete_output_path = view_dir_path + frame_id + '_%06d.jpg'

                        # The bellow should be on format '-vf scale=448:256'
                        ffmpeg_scale = 'scale='+ str(self.image_size[0]) + ':' + str(self.image_size[1])
    
                        ffmpeg_command = ['ffmpeg', '-ss', start_in_video, '-i', start_video_path, '-t',
                                          duration_ffmpeg, '-vf', ffmpeg_scale,
                                          '-r', str(self.frame_rate),
                                          complete_output_path, '-hide_banner']
                        print('\n')
                        print(ffmpeg_command)
                        print('\n')
                        subprocess.call(ffmpeg_command)
                        # Keep track of how much of the interval we have covered
                        duration_cumulative += remaining_duration
                        start = start + remaining_duration
                        clip_ind += 1
                # We went through all viewpoints, and move on to the next interval.        
                interval_ind += 1


    def create_clip_directories(self):
        subprocess.call(['mkdir', self.output_dir])
        for i, subject in enumerate(self.subjects):
            subject_dir_path = self.get_subject_dir_path(subject)
            subprocess.call(['mkdir', subject_dir_path])

            print('\n')
            print("Creating clip directories for subject {}...".format(subject))

            horse_df = self.data_selection_df.loc[self.data_selection_df['Subject'] == subject]
            for ind, row in horse_df.iterrows():
                start = str(row['Start'])
                end = str(row['End'])
                interval_dir_path = self.get_interval_dir_path(subject_dir_path, start, end)
                subprocess.call(['mkdir', interval_dir_path])
                for view in self.views:
                    view_dir_path = self.get_view_dir_path(interval_dir_path, view)
                    subprocess.call(['mkdir', view_dir_path])

    def _find_video_and_its_duration(self, subject, view, time):
        """
        subject: str (e.g. Aslan)
        view: int (e.g. 0, use lookup table)
        time: pd.datetime
        returns video path str, remainder duration pd.TimeDelta,
                start time in video (str)
        """

        # The videos are found in the following dir structure:
        # subject/yyyy-mm-dd/ch0x_yyyymmddHHMMSS.mp4
        # where x is the camera ID for that horse, found in viewpoints.csv

        subject_path = self.data_path + subject + '/'
        lookup_viewpoint = pd.read_csv('../data/viewpoints.csv', index_col='Subject')
        camera = lookup_viewpoint.at[subject, str(view)]
        camera_str = 'ch0' + str(camera)

        date_dir = [dd for dd in os.listdir(subject_path)
                    if pd.to_datetime(dd.split('/')[-1]).date() == time.date()]
        date_path = subject_path + date_dir[0] + '/'
       
        # Get list of all filenames on format 'ch0x_yyyymmddHHMMSS.mp4' 
        camera_filenames = [fn for fn in os.listdir(date_path)
                            if fn.startswith(camera_str) and '.mp4' in fn]

        # Remove .mp4 extension, format now 'ch0x_yyyymmddHHMMSS'
        camera_times = [fn[:fn.rindex('.')] for fn in camera_filenames]

        # Get only the time part, format now 'yyyymmddHHMMSS'
        filename_times = [fn.split('_')[-1] for fn in camera_times
                          if len(fn) == LEN_FILE_ID]

        # Convert strings to pd.TimeStamps
        time_stamps = list(map(from_filename_time_to_pd_datetime, filename_times))

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

        file_path = date_path + camera_filenames[correct_index]
        return file_path, remaining_duration, start_time_in_video


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

