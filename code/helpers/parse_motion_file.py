import os
from . import util
import numpy as np
import glob
from datetime import datetime
import multiprocessing
import pandas as pd


def parse_event(args):
	lines, idx_curr = args
	rel_lines = lines[idx_curr:idx_curr+9]
	date_str = ' '.join(rel_lines[1].split()[1:])
	datetime_object = datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')

	assert rel_lines[3].startswith('Major')
	assert rel_lines[4].startswith('Minor')
	assert rel_lines[5].startswith('Local')
	assert rel_lines[6].startswith('Host')
	assert rel_lines[7].startswith('Parameter')
	if not rel_lines[8].startswith('Camera'):
		return None

	vals = []
	for idx in [3,4,8]:
		line_curr = rel_lines[idx]
		val = line_curr[line_curr.index(':')+1:].strip()
		vals.append(val)

	return [datetime_object]+vals
	
def read_motion_file(file_curr):
	lines = util.readLinesFromFile(file_curr)
	lines = [line.strip('\r') for line in lines]
	lines = np.array(lines)
	idx_event = np.where(lines=='----------------------------')[0]
	idx_event = idx_event[::2]
	p = multiprocessing.Pool()
	args = [(lines, idx_curr) for idx_curr in idx_event]
	events = p.map(parse_event, args)
	events = [event for event in events if event is not None]
	columns = np.array(['Date Time','Major Type', 'Minor Type', 'Camera'])
	df = pd.DataFrame(columns = columns)
	for idx in range(len(events)):
		df.loc[idx]=events[idx]
	
	return df
	
	
	
	

def main():
	in_dir = '../data/tester_kit/naughty_but_nice'
	motion_files = glob.glob(os.path.join(in_dir,'*.txt'))
	motion_files.sort()
	motion_file = motion_files[2]
	print (motion_file)
	df = read_motion_file(motion_file)
	print (df[:10])


if __name__=='__main__':
	main()