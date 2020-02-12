import pandas as pd
import numpy as np
import torch
import os

class MultiViewFrameExtractor():
    def __init__(self, width, height, frame_rate, output_dir,
                 views, data_selection_path):
        """
        Args:
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

        self.image_size = (width, height)
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.views = views
        self.data_selection_df = pd.read_csv(data_selection_path)

        self.subjects = self.data_selection_df.Subject.unique()


    def extract_frames(self):
        for i, subject in enumerate(self.subjects):
            print("Extracting frames for subject {}...".format(subject))
            counter = 1  # Counter of clips from the same video.
            horse_df = self.data_selection_df.loc[self.data_selection_df['Subject'] == subject]
    
            for ind, row in horse_df.iterrows():
                print(row)
                occurences = check_if_unique_in_df(row['Video_ID'], horse_df)
    
                if occurences == 1:
                    clip_dir_path = output_dir + '/' + subject + '/' + row['Video_ID']
                elif occurences > 1:
                    clip_dir_path = output_dir + '/' + subject + '/' + row['Video_ID'] + \
                                    '_' + str(counter)
                    if counter == occurences:
                        counter = 1  # Reset
                    else:
                        counter += 1
                else:
                    print('Warning, 0 or negative occurences of clip')
    
                start = str(row['Start'])
                end = str(row['End'])
                # Remove "0 days " in the beginning of the timedelta to fit the ffmpeg command.
                length = str((pd.to_datetime(row['End']) - pd.to_datetime(row['Start'])))[7:]
                print(start, end, length)
    
                complete_output_path = clip_dir_path + '/frame_%06d.jpg'
                video_path = get_video_path(row)
    
                ffmpeg_command = ['ffmpeg', '-ss', start, '-i', video_path, '-t', length, '-vf',
                     'scale=448:256', '-r', str(1), complete_output_path, '-hide_banner']
                print(ffmpeg_command)
                subprocess.call(ffmpeg_command)


    def create_clip_directories(self):
        subprocess.call(['mkdir', self.output_dir])
        for i, subject in enumerate(self.subjects):
            subprocess.call(['mkdir', output_dir + '/' + subject])
            print("Creating clip directories for subject {}...".format(subject))
            counter = 1  # Counter of clips from the same video.
            horse_df = self.data_selection_df.loc[self.data_selection_df['Subject'] == subject]
            for vid in horse_df['Video_ID']:
                occurences = check_if_unique_in_df(vid, df)
                if occurences == 1:
                    clip_dir_path = output_dir + '/' + subject + '/' + vid
                elif occurences > 1:
                    clip_dir_path = output_dir + '/' + subject + '/' + vid + '_' + str(counter)
                    if counter == occurences: 
                        counter = 1  # Reset
                    else:
                        counter += 1
                else:
                    print('Warning, 0 or negative occurences of clip')
                subprocess.call(['mkdir', clip_dir_path])



def check_if_unique_in_df(file_name, df):
    """
    param file_name: str
    param df: pd.DataFrame
    :return: int [nb occurences of sequences from the same video clip in the pd.DataFrame]
    """
    return len(df[df['Video_ID'] == file_name])


