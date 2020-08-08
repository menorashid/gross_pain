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

def get_counts_indices(sdf, subject, data_dir_path):
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
            frame_path = get_image_name(subject, interval_ind, interval, view, frame, data_dir_path)
            if not os.path.isfile(frame_path):
                missing_inds.append(ind)
                subject_counter += 1
    return subject_counter, missing_inds 

def reduce_csvs(data_dir_path):
    subjects = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
    subject_counters = []
    subject_dfs = []
    subject_missing_inds = []
    for subject in subjects:
        
        print('Subject: ', subject)
        sdf = pd.read_csv(os.path.join(data_dir_path, subject + '_frame_index.csv'))
        
        subject_counter, missing_inds = get_counts_indices(sdf, subject, data_dir_path)
                    
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
        reduced_dfs[ind].to_csv(os.path.join(data_dir_path, subject + '_reduced_frame_index.csv'))

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

def main():
    data_dir_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps'
    reduce_csvs(data_dir_path)

        
        


if __name__=='__main__':
    main()