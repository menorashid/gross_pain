import os
import numpy as np
import glob
from datetime import datetime
import multiprocessing
import pandas as pd
from helpers import util

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
    enter_times = stable_df.loc[(stable_df['Event']==entrance_str) ,['Date Time']]
    enter_times_arr = np.array(enter_times.iloc[:,0])

    # get all people leaving times
    exit_times = stable_df.loc[(stable_df['Event']==exit_str) | (stable_df['Event']==camera_off_str),['Date Time']]
    exit_times_arr = np.array(exit_times.iloc[:,0])

    print ('\t'.join(['enter_time','exit_time','duration']))
    for enter_time in enter_times_arr:
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
def get_no_people_intervals(stable_df):
    
    events = np.array(stable_df['Event'])
    entrance_str = 'the team enters the stable'
    exit_str = 'the stable is empty'
    camera_on_str = 'all cameras are turned on'
    camera_off_str = 'at least one camera is off'

    # get all people entering times
    enter_times = stable_df.loc[(stable_df['Event']==entrance_str) | (stable_df['Event']==camera_off_str),['Date Time']]
    enter_times_arr = np.array(enter_times.iloc[:,0])

    # get all people leaving times
    exit_times = stable_df.loc[(stable_df['Event']==exit_str) ,['Date Time']]
    exit_times_arr = np.array(exit_times.iloc[:,0])

    # for each leaving time, find next enter time or end. these are empty stable segments
    no_ppl_intervals = []
    for idx_exit_time,exit_time in enumerate(exit_times_arr):

        enter_time,_ = get_next_closest_time(exit_time, enter_times_arr)
        no_ppl_intervals.append([exit_time, enter_time])

    return np.array(no_ppl_intervals)

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
    return next_closest_time,min_idx

def get_prev_closest_time(query_time, time_list):
    """ query_time: np.datetime64
        (the central bl or pain time (read from file),
        around which we take two hours: one before and one after.)

        time_list: np.array(np.datetime64)
        the end times of the no_people_intervals array
        (no_people_intervals[:,1])
        
        returns the closest previous time when a no-ppl (usable) interval ENDED."""

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
    return prev_closest_time,max_idx


# finds and trims intervals of time from a list, that are closest to query_time,  and have total duration of the required time length
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
            if end > injection_time:
                print(end, injection_time)
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
    csv_path = '../data/frame_extraction_files/overview_stable_nbn_correction.csv'
    pain_times_file = '../data/frame_extraction_files/peak_pain.txt'
    injection_times_file = '../data/frame_extraction_files/injection_times.txt'
    bl_times_file = '../data/frame_extraction_files/pre_bl_cps_time.txt'

    out_file = '../data/frame_extraction_files/test_pain_no_pain_intervals_for_extraction.csv'

    # trimming 2 minutes around when people are in the stable 
    ppl_buffer = np.timedelta64(2,'m') 

    # we need two hours of data 
    required_time = np.timedelta64(2,'h')


    stable_df = get_stable_dataframe(csv_path)
   
    # no_people_intervals is an array of arrays, like so: [[start,end][start,end]...] 
    # where start and end are of type numpy.datetime64
    no_people_intervals = get_no_people_intervals(stable_df)
    
    # add 2 min buffer to make sure there are no people
    no_people_intervals[:,0] = no_people_intervals[:,0]+ppl_buffer
    no_people_intervals[:,1] = no_people_intervals[:,1]-ppl_buffer
    

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
        intervals_keep += get_aft_intervals(row['Date Time'], no_people_intervals,
                                            required_time, pain, injection_time)
        # intervals_keep += get_bef_intervals(row['Date Time'], no_people_intervals,
        #                                     required_time/2., pain, injection_time)
        csv_lines += format_intervals_for_csv(subject.lower().replace(' ','_'), str(pain), intervals_keep)

    csv_lines.sort()
    print (out_file,len(csv_lines))
    util.writeFile(out_file, csv_lines)


    
    



    




if __name__=='__main__':
    main()
