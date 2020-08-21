import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import glob
from datetime import datetime
import multiprocessing
import pandas as pd
from helpers import util

# Number of hours of data per class, per subject
# (=> total dataset will be NB_HOURS * 2 * 8 * 4)
# for two classes, eight subjects and four viewpoints.
NB_HOURS = 2
# trimming 2 minutes around when people are in the stable 
PPL_BUFFER = np.timedelta64(2,'m') 

# returns np.datetime64 time column, and comment about action
def get_stable_dataframe(csv_path):
    csv_df = pd.read_csv(csv_path)
    times = csv_df['Time'].values
    times = np.array(times)    
    columns = np.array(['Date Time','Event'])
    df = pd.DataFrame(columns = columns)
    for row_idx,row in csv_df.iterrows():
        
        [date_curr,time_curr,event] = [row[key] for key in ['Date','Time','Event']]
        if str(date_curr)=='nan':
            continue

        if 'Night' in time_curr:
            continue
        
        event = event.strip().lower()
        
        date_str = date_curr+' '+time_curr
        datetime_object = datetime.strptime(date_str,'%m/%d/%y %H:%M:%S')   

        df.loc[len(df)]=[datetime_object,event]

    return df
        
# makes a df from a file like peak_pain.txt
def get_observation_df(obs_path,pain = 0):
    lines = util.readLinesFromFile(obs_path)
    lines = [line.split(',') for line in lines]
    columns = np.array(['Horse','Date Time','Pain'])
    df = pd.DataFrame(columns = columns)
    for line in lines:
        (horse, date_curr, time_curr) = line
        datetime_object = datetime.strptime(date_curr+' '+time_curr,'%m/%d/%y %H:%M:%S')   
        df.loc[len(df)]=[horse,datetime_object, pain]
    return df

# prints out enter and exit times for people in stable
def sanity_check_times(stable_df):
    events = np.array(stable_df['Event'])
    entrance_str = 'the team enters the stable'
    exit_str = 'the stable is empty'
    camera_off_str = 'at least one camera is off'
    # get all people entering times
    enter_or_camera_off_times = stable_df.loc[(stable_df['Event']==entrance_str) ,['Date Time']]
    enter_or_camera_off_times_arr = np.array(enter_or_camera_off_times.iloc[:,0])

    # get all people leaving times
    exit_times = stable_df.loc[(stable_df['Event']==exit_str) | (stable_df['Event']==camera_off_str),['Date Time']]
    exit_times_arr = np.array(exit_times.iloc[:,0])

    print ('\t'.join(['enter_time','exit_time','duration']))
    for enter_time in enter_or_camera_off_times_arr:
        exit_time,_ = get_next_closest_time(enter_time,exit_times_arr)
        min_diff = int(exit_time - enter_time)

        if min_diff>8.64e+13:
            duration_str = 'longer than a day'
        else:
            duration_str = pd.to_datetime(np.timedelta64(min_diff).astype('datetime64[D]')).strftime('%H:%M:%S')

        enter_time_str = pd.to_datetime(enter_time).strftime('%m/%d/%y %H:%M:%S')
        exit_time_str = pd.to_datetime(exit_time).strftime('%m/%d/%y %H:%M:%S')
        print ('\t'.join([enter_time_str,exit_time_str,duration_str]))

# gets nx2 np datetime array of times when the stable is empty and cameras are on
def get_no_people_camera_on_intervals(stable_df):
    
    events = np.array(stable_df['Event'])
    entrance_str = 'the team enters the stable'
    exit_str = 'the stable is empty'
    camera_off_str = 'at least one camera is off'
    camera_on_str = 'all cameras are turned on'

    # get all people entering times
    enter_times = stable_df.loc[(stable_df['Event']==entrance_str)]
    enter_times_arr = np.array(enter_times.iloc[:,0])

    # get all camera off times
    camera_off_times = stable_df.loc[(stable_df['Event']==camera_off_str),['Date Time']]
    camera_off_arr = np.array(camera_off_times.iloc[:,0])

    # get all people leaving times
    exit_times = stable_df.loc[(stable_df['Event']==exit_str) ,['Date Time']]
    exit_times_arr = np.array(exit_times.iloc[:,0])
    
    # get all camera on times
    camera_on_times = stable_df.loc[(stable_df['Event']==camera_on_str),['Date Time']]
    camera_on_arr = np.array(camera_on_times.iloc[:,0])

    # find the intersection between empty stable segments and times when the cameras were on
    no_ppl_camera_on_intervals = []
    for idx_exit_time, exit_time in enumerate(exit_times_arr):
        enter_time, _ = get_next_closest_time(exit_time, enter_times_arr)

        # check that camera is on at the beginning of the interval
        camera_on = check_if_camera_is_on(exit_time,
                                          camera_on_arr,
                                          camera_off_arr)
        if not camera_on:
            # pick the next time the camera is on as new start of interval ('exit time') instead
            exit_time, _ = get_next_closest_time(exit_time, camera_on_arr)
            # but make sure it does not happen after the next enter time
            if exit_time >= enter_time:
                continue
        # the camera might turn on and off during the empty stable interval
        on_times_in_interval = get_all_on_times_in_interval(exit_time,
                                                            enter_time,
                                                            camera_on_arr)
        on_times = [exit_time] + on_times_in_interval
        for on_time in on_times:
            # want to find all usable sub-intervals before the team enters again
            next_off_time, _ = get_next_closest_time(on_time, camera_off_arr)
            if next_off_time < enter_time:
                interval = [on_time, next_off_time]
            else:
                interval = [on_time, enter_time]
            # Intervals need to be longer than twice the people buffer
            if interval[1]-interval[0] > 2*PPL_BUFFER:
                no_ppl_camera_on_intervals.append(interval)

    return np.array(no_ppl_camera_on_intervals)


def get_all_on_times_in_interval(start, end, on_times):
    on_times_in_interval = [ot for ot in on_times if (ot > start and ot < end)] 
    return on_times_in_interval


def check_if_camera_is_on(query_time, on_times, off_times):
    last_camera_on, _ = get_prev_closest_time(query_time, on_times)
    last_camera_off, _ = get_prev_closest_time(query_time, off_times)
    if last_camera_on and (last_camera_off is None):
        return True
    # np.datetime64 > None evaluates to True (if there is no prev off-time)
    return last_camera_on > last_camera_off
    
    
# these two find the the time thats closest to query time from a list
def get_next_closest_time(query_time, time_list):
    diffs = time_list - query_time
    diffs = diffs.astype(int)
    # We are only interested in the positive times,
    # so we set the negative to max to rule them out
    diffs[diffs<0] = np.iinfo(type(diffs[0])).max
    min_idx = np.argmin(diffs)
    min_diff = diffs[min_idx]
    next_closest_time = time_list[min_idx]
    return next_closest_time, min_idx


def get_prev_closest_time(query_time, time_list):
    """ query_time: np.datetime64
        (the central bl or pain time (read from file),
        around which we take two hours: one before and one after.)

        time_list: np.array(np.datetime64)
        return: (np.datetime64, int), or (None, None) if there is none.

        example:
        the time_list may be the end times of the no_people_intervals array
        (no_people_intervals[:,1]). in that case the method returns the closest
        previous time when a no-ppl (usable) interval ENDED."""

    diffs = time_list - query_time
    diffs = diffs.astype(int)
    # We are only interested in the negative times,
    # so we set the positive to min to rule them out
    diffs[diffs>0] = np.iinfo(type(diffs[0])).min
    # Now get the max out of the negative ones (i.e. the smallest diff,
    # i.e. closest in time before query_time)
    max_idx = np.argmax(diffs)
    max_diff = diffs[max_idx]
    prev_closest_time = time_list[max_idx]
    if prev_closest_time >= query_time:  # If there was no previous time.
        prev_closest_time = None
        max_idx = None
    return prev_closest_time, max_idx


# finds and trims intervals of time from a list, that are closest to query_time,
# and have total duration of the required time length
def get_aft_intervals(query_time, no_people_intervals, req_time, pain, injection_time):
    intervals_keep = []
    query_time = np.datetime64(query_time)
    injection_time = np.datetime64(injection_time)
    total_time = np.timedelta64(0,'h')

    while total_time<req_time:
        _,next_idx = get_next_closest_time(query_time,no_people_intervals[:,0])

        start = no_people_intervals[next_idx,0]
        end = no_people_intervals[next_idx,1]
        if pain:
            assert(start > injection_time)
        else:
            assert(end < injection_time)

        intervals_keep.append(no_people_intervals[next_idx,:])
        total_time += no_people_intervals[next_idx,1]-no_people_intervals[next_idx,0]
        query_time = no_people_intervals[next_idx,1]

    time_to_shave = total_time - req_time
    intervals_keep[-1][1] = intervals_keep[-1][1] - time_to_shave
    return intervals_keep

def get_bef_intervals(query_time, no_people_intervals, req_time, pain, injection_time):
    intervals_keep = []
    query_time = np.datetime64(query_time)
    injection_time = np.datetime64(injection_time)
    total_time = np.timedelta64(0,'h')

    while total_time<req_time:
        _,prev_idx = get_prev_closest_time(query_time,no_people_intervals[:,1])

        start = no_people_intervals[prev_idx,0]
        end = no_people_intervals[prev_idx,1]
        if pain:
            assert(start > injection_time)
        else:
            assert(end < injection_time)

        intervals_keep.append(no_people_intervals[prev_idx,:])
        total_time += no_people_intervals[prev_idx,1]-no_people_intervals[prev_idx,0]
        query_time = no_people_intervals[prev_idx,0]

    time_to_shave = total_time - req_time
    intervals_keep[-1][0] = intervals_keep[-1][0] + time_to_shave
    return intervals_keep


def format_intervals_for_csv(horse, pain, intervals_keep):
    csv_lines = []
    for interval_curr in intervals_keep:
        csv_line = []
        csv_line.append(horse)
        csv_line.append(pd.to_datetime(interval_curr[0]).strftime('%Y%m%d%H%M%S'))
        csv_line.append(pd.to_datetime(interval_curr[1]).strftime('%Y%m%d%H%M%S'))
        csv_line.append(pain)
        csv_line = ','.join(csv_line)
        csv_lines.append(csv_line)
    return csv_lines

def main():
    save_dir = '../data/frame_extraction_files/'
    csv_path =  os.path.join(save_dir, 'overview_stable_nbn_correction.csv')
    pain_times_file = os.path.join(save_dir, 'peak_pain.txt')
    injection_times_file = os.path.join(save_dir, 'injection_times.txt')
    bl_times_file = os.path.join(save_dir, 'pre_bl_cps_time.txt')

    out_file = os.path.join(save_dir, 'pain_no_pain_x{}h_intervals_for_extraction.csv'.format(NB_HOURS))

    required_time = np.timedelta64(NB_HOURS,'h')

    stable_df = get_stable_dataframe(csv_path)
   
    # no_people_intervals is an array of arrays, like so: [[start,end][start,end]...] 
    # where start and end are of type numpy.datetime64
    no_people_intervals = get_no_people_camera_on_intervals(stable_df)
    
    # add PPL_BUFFER min buffer to make sure there are no people
    no_people_intervals[:,0] = no_people_intervals[:,0]+PPL_BUFFER
    no_people_intervals[:,1] = no_people_intervals[:,1]-PPL_BUFFER
    

    pain_times_df = get_observation_df(pain_times_file,1)
    bl_times_df = get_observation_df(bl_times_file,0)

    injection_times_df = get_observation_df(injection_times_file,1)

    csv_lines = []

    for idx_row,row in pain_times_df.iterrows():
        intervals_keep = []
        subject = row['Horse']
        pain = row['Pain']
        injection_time = injection_times_df.iloc[idx_row]['Date Time']
        # get half required time after pain time
        intervals_keep += get_aft_intervals(row['Date Time'], no_people_intervals,
                                            required_time/2., pain, injection_time)
        # get half required time before pain time
        intervals_keep += get_bef_intervals(row['Date Time'], no_people_intervals,
                                            required_time/2., pain, injection_time)
        # format for file
        csv_lines += format_intervals_for_csv(subject.lower().replace(' ','_'), str(pain), intervals_keep)


    for idx_row,row in bl_times_df.iterrows():
        intervals_keep = []
        subject = row['Horse']
        pain = row['Pain']
        injection_time = injection_times_df.iloc[idx_row]['Date Time']
        # get required time after baseline time
        # take 2h after bl because the before time is sometimes dark in the morning.
        intervals_keep += get_aft_intervals(row['Date Time'], no_people_intervals,
                                            required_time, pain, injection_time)
        # intervals_keep += get_bef_intervals(row['Date Time'], no_people_intervals,
        #                                     required_time/2., pain, injection_time)
        csv_lines += format_intervals_for_csv(subject.lower().replace(' ','_'), str(pain), intervals_keep)


    # Double check that there are no duplicates
    starts = [s for s in list(no_people_intervals[:,0])]
    ends = [e for e in list(no_people_intervals[:,1])]
    assert(len(starts) == len(set(starts)))
    assert(len(ends) == len(set(ends)))

    csv_lines.sort()
    csv_lines = ['subject,start,end,pain'] + csv_lines

    print (out_file,len(csv_lines))
    util.writeFile(out_file, csv_lines)


    




if __name__=='__main__':
    main()
