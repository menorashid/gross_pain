import os
import glob
import numpy as np
from helpers import util, visualize

def view_frames_side_by_side(meta_dir, out_file_html, str_replace):

	dirs_full = [dir_curr for dir_curr in glob.glob(os.path.join(meta_dir,'*')) if os.path.isdir(dir_curr)]
	im_lists = []
	for dir_curr in dirs_full:
		im_list = list(glob.glob(os.path.join(dir_curr,'*')))
		im_list.sort()
		im_lists.append(im_list)

	num_ims = [len(im_list) for im_list in im_lists]
	im_rows = []
	caption_rows = []
	for row_num in range(np.min(num_ims)):
		im_row = []
		caption_row = []
		for im_list,dir_curr in zip(im_lists,dirs_full):
			im_curr = im_list[row_num]
			im_name = os.path.split(im_curr)[1]
			caption_curr = ' '.join([os.path.split(dir_curr)[1],im_name])
			
			im_row.append(im_curr.replace(str_replace[0],str_replace[1]))
			caption_row.append(caption_curr)

		im_rows.append(im_row)
		caption_rows.append(caption_row)

	visualize.writeHTML(out_file_html,im_rows,caption_rows,height=256,width=448)


def view_multiple_dirs(dirs_to_check, out_dir_html, str_replace = ['..','/gross_pain']):
	for dir_curr in dirs_to_check:
		out_file_html = '_'.join(dir_curr.split('/')[-2:])+'.html'
		out_file_html = os.path.join(out_dir_html, out_file_html)
		view_frames_side_by_side(dir_curr, out_file_html,  str_replace)
		print (out_file_html)

def main():

	# dirs_to_check = util.readLinesFromFile('../scratch/dirs_to_check.txt')
	# dirs_to_check = [dir_curr.strip('/') for dir_curr in dirs_to_check]
	# out_dir_html = '../data/frames_to_check_for_humans_html'
	# util.mkdir(out_dir_html)

	# dirs_to_check = ['../data/frames_to_check_for_humans_debug/julia/20190328124424_134537']
	# out_dir_html = '../data/frames_to_check_for_humans_debug_html'
	
	pass

	






if __name__=='__main__':
	main()