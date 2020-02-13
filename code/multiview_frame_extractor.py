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

                    video = self.find_video(subject, view, start)

                    # Identify the video containing the start
                    # Check if it also contains the end
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
                import pdb; pdb.set_trace()
                start = str(row['Start'])
                end = str(row['End'])
                interval_dir_path = self.get_interval_dir_path(subject_dir_path, start, end)
                subprocess.call(['mkdir', interval_dir_path])
                for view in self.views:
                    view_dir_path = self.get_view_dir_path(interval_dir_path, view)
                    subprocess.call(['mkdir', view_dir_path])

    def find_video(self, subject, view, time):
        """
        subject: str (e.g. Aslan)
        view: int (e.g. 0, use lookup table)
        time: pd.datetime
        returns video path str
        """
        subject_path = self.data_path + subject + '/'
        lookup_viewpoint = pd.read_csv('../data/viewpoints.csv', index_col='Subject')
        camera = lookup_viewpoint.at[subject, str(view)]
        camera_str = 'ch0' + str(camera)

        
        date_dir = [dd for dd in os.listdir(subject_path)
                    if pd.to_datetime(dd.split('/')[-1]).date() == time.date()]
        date_path = subject_path + date_dir[0] + '/'
        
        camera_filenames = [fn[:fn.rindex('.')] for fn in os.listdir(date_path)
                            if fn.startswith(camera_str) and '.mp4' in fn]

        filename_times = [fn.split('_')[-1] for fn in camera_filenames]

        time_stamps = map(from_filename_time_to_pd_datetime, filename_times)

        # video_filename = 
        import pdb; pdb.set_trace()

        for path, dirs, files in sorted(os.walk(subject_path)):
            date = pd.to_datetime(path.split('/')[-1]).date()
            print(path)

        return 'subj_path/date/video_path.mp4'
        



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

