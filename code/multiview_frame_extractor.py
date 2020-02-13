import pandas as pd
import numpy as np
import subprocess
import torch
import os


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

    def get_subject_dir_path(self, subject):
        return self.output_dir + '/' + subject + '/'

    def get_interval_dir_path(self, subject_dir_path, start, end):
        return subject_dir_path + start + '_' + end + '/'

    def get_view_dir_path(self, interval_dir_path, view):
        return interval_dir_path + str(view) + '/'

    def extract_frames(self):
        for i, subject in enumerate(self.subjects):
            subject_dir_path = self.get_subject_dir_path(subject)
            print("Extracting frames for subject {}...".format(subject))
            counter = 1  # Counter of clips from the same video.
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
                start = pd.to_datetime(row['Start'], format='%Y%m%d%H%M%S')
                end = pd.to_datetime(row['End'], format='%Y%m%d%H%M%S')
                length = str(start - end)

                for view in self.views:
                    view_dir_path = self.get_view_dir_path(interval_dir_path, view)

                    # Identify the video containing the start
                    start_video_path, duration = self.find_video_and_its_duration(subject, view, start)
                    import pdb; pdb.set_trace()

                    # Check if it also contains the end (bool)
                    if duration < (end-start):
                        print('Need to look for next clip.')
                        
                    # same_video = self.timestamp_in_video(start_video_path, end)
                      # If yes -- extract until end of interval, done.
                      # If no -- extract until end of video,
                      #          go to next video,
                      #          repeat.


                    # # Remove "0 days " in the beginning of the timedelta to fit the ffmpeg command.
                    # length_ffmpeg = length[7:]
                    # print(start, end, length)
    
                    # complete_output_path = clip_dir_path + '/frame_%06d.jpg'
                    # video_path = get_video_path(row)
    
                    # ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-t', length, '-vf',
                    #      'scale=448:256', '-r', str(1), complete_output_path, '-hide_banner']
                    # print(ffmpeg_command)
                    # subprocess.call(ffmpeg_command)


    def create_clip_directories(self):
        subprocess.call(['mkdir', self.output_dir])
        for i, subject in enumerate(self.subjects):
            subject_dir_path = self.get_subject_dir_path(subject)
            subprocess.call(['mkdir', subject_dir_path])

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

    def find_video_and_its_duration(self, subject, view, time):
        """
        subject: str (e.g. Aslan)
        view: int (e.g. 0, use lookup table)
        time: pd.datetime
        returns video path str
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
        filename_times = [fn.split('_')[-1] for fn in camera_times]
        # Convert strings to pd.TimeStamps
        time_stamps = list(map(from_filename_time_to_pd_datetime, filename_times))
        # Get the pd.TimeDeltas for each timestamp in the list,
        # relative to the given time
        time_deltas = [get_timedelta(time, ts) for ts in time_stamps]

        # Get the index before that of the first negative TimeDelta
        correct_index = get_index_for_correct_file(time_deltas)

        # Also return the length of the clip
        duration = time_stamps[correct_index+1] - time_stamps[correct_index]

        file_path = date_path + camera_filenames[correct_index]
        return file_path, duration

    def timestamp_in_video(self, video_path, time):
        """
        video_path: str
        time: pd.TimeStamp
        return: bool
        """
        time_str = video_path[:video_path(rindex('.'))].split('_')[-1]
        time_stamp = from_filename_time_to_pd_datetime(time_str)

        


def get_index_for_correct_file(time_deltas):
    """
    time_deltas: [pd.TimeDelta] list of timedeltas.
    Finds index of the file we're looking for.
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


def check_if_unique_in_df(file_name, df):
    """
    param file_name: str
    param df: pd.DataFrame
    :return: int [nb occurences of sequences from the same video clip in the pd.DataFrame]
    """
    return len(df[df['Video_ID'] == file_name])


def get_video_path(row):
    p = 'Pain' if row['Pain']==1 else 'No_Pain'
    path = row['Subject'] + '/' + p + '/' + row['Video_ID'] + '.mp4'
    return path

def from_filename_time_to_pd_datetime(yyyymmddHHMMSS):
    return pd.to_datetime(yyyymmddHHMMSS, format='%Y%m%d%H%M%S')

