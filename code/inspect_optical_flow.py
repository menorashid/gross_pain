from scipy.misc import imsave
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import sys
import os

import utils.flow_utils as flow_utils


def read_flo_and_save_as_jpg(flow_dir):
    total_mag = 0
    for filename in os.listdir(flow_dir):
        if filename.endswith('.flo'):
            total_path = flow_dir + filename
            flow = flow_utils.readFlow(total_path)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            total_mag += mag
            mag = np.reshape(mag, (mag.shape[0], mag.shape[1], 1))
            flow = np.concatenate((flow, mag), axis=2)
            flow = cv2.normalize(flow, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
            fn_stem = total_path[:-4]
            imsave(fn_stem + '.jpg', flow)
        else:
            continue
    # print('Total optical flow magnitude for {}: {}'.format(flow_dir,
    #                                               np.sum(total_mag)))


def get_flow_magnitude_for_one_clip(flow_dir, encoding='.jpg'):
    total_mag = 0
    for filename in os.listdir(flow_dir):
        if filename.endswith(encoding):
            total_path = flow_dir + filename
            if encoding=='.flo':
                flow = flow_utils.readFlow(total_path)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            else:
                flow = cv2.imread(total_path)
                mag = flow[:,:,2]
            total_mag += mag
        else:
            continue
    # print('Total optical flow magnitude for {}: {}'.format(flow_dir,
    #                                               np.sum(total_mag)))
    return total_mag


def get_column_sum(df, column='Flow_Magnitude'):
    return df[column].sum()


def get_number_of_entries(df):
    return len(df)

def get_nb_frames_above_threshold(df, threshold, column='Flow_Magnitude'):
    array = df[column].values
    return sum(i > threshold for i in array)


def print_flow_magnitude_per_pain_label(dfp, dfnp):
    magp = get_column_sum(dfp)
    magnp = get_column_sum(dfnp)
    print('Flow magnitude for pain: {}\n'.format(magp))
    print('Flow magnitude for no pain: {}\n'.format(magnp))

    print('NORMALIZED\n')
    nb_frames_p = get_number_of_entries(dfp)
    nb_frames_np = get_number_of_entries(dfnp)

    norm_p = magp/nb_frames_p
    norm_np = magnp/nb_frames_np

    print('Normalized flow magnitude for pain: {}\n'.format(norm_p))
    print('Normalized flow magnitude for no pain: {}\n'.format(norm_np))

def print_nb_frames_with_movement(dfp, dfnp, threshold):
    nb_frames_with_movement_p = get_nb_frames_above_threshold(dfp, threshold)
    nb_frames_with_movement_np = get_nb_frames_above_threshold(dfnp, threshold)
    print('Pain, nb frames above {}: {}\n'.format(threshold, nb_frames_with_movement_p))
    print('No pain, nb frames above {}: {}\n'.format(threshold, nb_frames_with_movement_np))


def print_jpg_magnitude_per_pain_label(df, flow_dir):
    time_format = '%H:%M:%S'
    subjects = df.Subject.unique()
    for subject in subjects:
        pain_mag = 0
        nopain_mag = 0
        horse_df = df.loc[df['Subject'] == subject]
        for ind, row in horse_df.iterrows():
            subfolders = '/inference/run.epoch-0-flow-field/'
            flow_dir_clip = flow_dir + row['Subject'] + '/' + row['Video_ID']
            flow_dir_clip += subfolders
            mag = get_flow_magnitude_for_one_clip(flow_dir_clip)
            length = (datetime.strptime(row['End'], time_format) - datetime.strptime(row['Start'], time_format)).seconds
            if row['Pain'] == 1:
                pain_mag += mag/length
            elif row['Pain'] == 0:
                nopain_mag += mag/length
            else:
                print('Warning, missing pain label...')
        print('Results for {}'.format(subject))
        print('Total optical flow magnitude for the pain label: {}'.format(np.sum(pain_mag)))
        print('Total optical flow magnitude for the no pain label: {}'.format(np.sum(nopain_mag)))



if __name__ == '__main__':
    if len(sys.argv) > 1:
        flow_dir = sys.argv[1]
    # df = pd.read_csv('videos_cleaner_overview.csv')
    # print_jpg_magnitude_per_pain_label(df, flow_dir)

    dfp = pd.read_csv('flow_magnitudes_pain.csv')
    dfnp = pd.read_csv('flow_magnitudes_nopain.csv')
    print_flow_magnitude_per_pain_label(dfp, dfnp)
    thresholds = range(1000,30000, 1000)
    for t in thresholds:
        print_nb_frames_with_movement(dfp, dfnp, threshold=t)

