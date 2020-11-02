import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_image_name(subject, interval_ind, interval, view, frame, data_dir_path):
    frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
                             str(view), '%06d'%frame])
    return os.path.join(data_dir_path,'{}/{}/{}/{}.jpg'.format(subject,
                                                        interval,
                                                        view,
                                                        frame_id))

def get_flow_name(subject, interval_ind, interval, view, frame, data_dir_path):
    frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
                             str(view), '%06d'%frame])
    return os.path.join(data_dir_path,'{}/{}/{}_opt/{}.png'.format(subject,
                                                        interval,
                                                        view,
                                                        frame_id))

def get_counts_indices(sdf, subject, data_dir_path, flow=False):
    subject_counter = 0
    sdf_length = len(sdf)
    missing_inds = []
    with tqdm(total=sdf_length) as pbar:
        for ind, row in sdf.iterrows():
            pbar.update(1)
            interval_ind = row['interval_ind']
            interval = row['interval']
            view = row['view']
            frame = row['frame']
            if not flow:
                frame_path = get_image_name(subject, interval_ind, interval, view, frame, data_dir_path)
            else:
                frame_path = get_flow_name(subject, interval_ind, interval, view, frame, data_dir_path)
            if not os.path.isfile(frame_path):
                missing_inds.append(ind)
                subject_counter += 1
    return subject_counter, missing_inds 

def reduce_csvs(data_dir_path, str_aft = '_frame_index.csv',flow = False):
    subjects = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
    subject_counters = []
    subject_dfs = []
    subject_missing_inds = []
    for subject in subjects:
        
        print('Subject: ', subject)
        sdf = pd.read_csv(os.path.join(data_dir_path, subject + str_aft))
        
        subject_counter, missing_inds = get_counts_indices(sdf, subject, data_dir_path, flow)
                    
        subject_missing_inds.append(missing_inds)
        subject_counters.append(subject_counter)
        subject_dfs.append(sdf)
        
    print ('subject_counters',subject_counters)
    # print (subject_dfs[0])

    reduced_dfs = []
    for i in range(len(subject_dfs)):
        sdf = subject_dfs[i]
        reduced = sdf.drop((sdf.index[subject_missing_inds[i]]))
        reduced = reduced.reset_index(drop=True)
        reduced_dfs.append(reduced)

    print ([(len(subject_dfs[i]) - len(reduced_dfs[i])) == subject_counters[i] for i in range(len(subject_counters))])

    # print (reduced_dfs[0][subject_missing_inds[0][0]-1:subject_missing_inds[0][0]+1])

    for ind, subject in enumerate(subjects):
        subject_counter, missing_inds = get_counts_indices(reduced_dfs[ind],subject, data_dir_path)
        assert len(missing_inds)==0
        # print (reduced_dfs[ind])
        if flow:
            out_file = os.path.join(data_dir_path, subject + '_optreduced'+str_aft)
        else:
            out_file = os.path.join(data_dir_path, subject + '_reduced'+str_aft)
        print (out_file)
        reduced_dfs[ind].to_csv(out_file)

def sanity_check():
    subjects = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
    data_dir_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps'
    sofia_dir = '../data/sofia_reduced'
    import glob
    csvs = glob.glob(os.path.join(sofia_dir,'*.csv'))
    for csv_file in csvs:
        m_file = os.path.join(data_dir_path,os.path.split(csv_file)[1])
        data_all = []
        for file_curr in [csv_file, m_file]:
            with open(csv_file,'r') as f:
                csv_data=f.read()
                data_all.append(csv_data)
                print (len(csv_data))
                print (csv_data[:100])

        assert (data_all[0]==data_all[1])

def make_smaller_fps_csv(data_dir_path,old_csv_aft,new_csv_aft,jump_val, subjects = None, view_main = 0):
    if subjects is None:
        subjects = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']

    column_headers = ['interval', 'interval_ind', 'view', 'subject', 'pain', 'frame']

    for subject in subjects:
        csv_file = os.path.join(data_dir_path, subject+old_csv_aft)
        out_file = os.path.join(data_dir_path, subject+new_csv_aft)
        assert os.path.exists(csv_file)
        assert out_file is not csv_file
        
        frames = pd.read_csv(csv_file)
        rel_intervals = frames.interval.unique()
        rel_views = frames.view.unique()
        # print (rel_views)
        rows_keep_all = []
        total = 0
        for idx_interval,interval in enumerate(rel_intervals):
            rel_frames = frames.loc[(frames['interval'] == interval) & (frames['view'] == view_main)]
            rel_frames = rel_frames.sort_values('frame')
            frames_keep = rel_frames.frame.values
            frames_keep = frames_keep[::jump_val]
            # print (len(rel_frames),len(frames_keep))
            
            for view in rel_views:
                rel_frames = frames.loc[(frames['interval'] == interval) & (frames['view'] == view)]
                rel_frames = rel_frames.sort_values('frame')
                rows_keep = rel_frames.loc[np.in1d(rel_frames['frame'].values,frames_keep)]
                total+=len(rel_frames)
                # print (view, len(rel_frames),len(rows_keep),total)
                
                rows_keep_all.append(rows_keep)

        rows_keep_all = pd.concat(rows_keep_all)
        # print(rows_keep_all.columns.values)
        rows_keep_all = rows_keep_all.drop(columns = [rows_keep_all.columns.values[0]])
        rows_keep_all = rows_keep_all.reset_index(drop =True)

        # print (len(rows_keep_all), rows_keep_all[::10000])
        
        rows_keep_all.to_csv(path_or_buf= out_file)
        print (subject, len(frames), len(rows_keep_all), out_file)

def main():

    # data_dir_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps'
    # reduce_csvs(data_dir_path)

    data_dir_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    # str_aft = '_thresh_0.70_frame_index.csv'
    # str_aft = '_percent_0.01_frame_index.csv'
    # print (data_dir_path)
    # reduce_csvs(data_dir_path, str_aft = str_aft)

    str_aft = '_reduced_percent_0.01_frame_index.csv'
    print (data_dir_path)
    reduce_csvs(data_dir_path, str_aft = str_aft, flow = True)


    # data_dir_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps'
    # old_csv_aft = '_frame_index.csv'
    # new_csv_aft = '_2fps_frame_index.csv'
    # jump_val = 5
    # subjects = None
    # # make_smaller_fps_csv(data_dir_path,old_csv_aft,new_csv_aft,jump_val, subjects = subjects)
    # data_dir_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    # reduce_csvs(data_dir_path, str_aft = new_csv_aft)

        
        


if __name__=='__main__':
    main()