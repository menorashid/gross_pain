import pandas as pd
import numpy as np
import subprocess
import sys


def check_if_unique_in_df(file_name, df):
    """
    param file_name: str
    param df: pd.DataFrame
    :return: int [nb occurences of sequences from the same video clip]
    """
    return len(df[df['Video_ID'] == file_name])


def get_video_path(row):
    p = 'Pain' if row['Pain']==1 else 'No_Pain'
    path = row['Subject'] + '/' + p + '/' + row['Video_ID'] + '.mp4'
    return path


def create_clip_directories(df, subjects, output_dir):
    subprocess.call(['mkdir', output_dir])
    for i, subject in enumerate(subjects):
        subprocess.call(['mkdir', output_dir + '/' + subject])
        print("Creating clip directories for subject {}...".format(subject))
        counter = 1  # Counter of clips from the same video.
        horse_df = df.loc[df['Subject'] == subject]
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


def extract_frames(df, subjects, output_dir):
    for i, subject in enumerate(subjects):
        print("Extracting frames for subject {}...".format(subject))
        counter = 1  # Counter of clips from the same video.
        horse_df = df.loc[df['Subject'] == subject]

        for ind, row in horse_df.iterrows():
            print(row)
            occurences = check_if_unique_in_df(row['Video_ID'], horse_df)

            if occurences == 1:
                clip_dir_path = output_dir + '/' + subject + '/' + row['Video_ID']
            elif occurences > 1:
                clip_dir_path = output_dir + '/' + subject + '/' + row['Video_ID'] + '_' + str(counter)
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


if __name__ == '__main__':
    df = pd.read_csv('videos_cleaner_overview.csv')
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        print('Provide name of output directory.')
    subjects = df.Subject.unique()
    create_clip_directories(df, subjects, output_dir)
    extract_frames(df, subjects, output_dir)


