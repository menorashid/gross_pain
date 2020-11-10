from helpers import util
import glob
import os
import numpy as np
import re

def parse_test_file(test_file,num_select = -1, idx_select = -1):
	lines = util.readLinesFromFile(test_file)[1:]
	# line = lines[0]
	# print (line)
	# print (util.replaceSpecialChar(line,' ').split())
	# print ('NUM SELECT',num_select)
	
	# print (lines, test_file)
	# print (util.replaceSpecialChar(lines[0],' ').split())
	
	accus = np.array([float(util.replaceSpecialChar(line,' ').split()[num_select]) for line in lines])
	# print (accus)
	accus = accus*100
	idx_max = np.argmax(accus)
	end_accu = accus[-1]
	max_accu = accus[idx_max]
	# print (idx_max, max_accu)
	# input()
	if idx_select>-1:
		# idx_val = idx_select+1
		# print (len(accus))
		val_accu = accus[idx_select]
		return end_accu, len(accus)-1, max_accu, idx_max, val_accu, idx_select
	else:
		return end_accu, len(accus)-1, max_accu, idx_max

def main():
	has_val = False
	dirs_select = list(range(8))
	idx_metric = -1
	# meta_dir = '../output/pain_lstm_wbn_512_MIL_Loss_CE_painLSTM_512_1_seqlen_10_milce_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	# # meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_Mix_painLSTM_1024_1_seqlen_10_wbn_allout_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	# meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_CE_painLSTM_1024_1_seqlen_10_wbn_allout_milce_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	
	# meta_dir = '../output/pain_lstm_wln_MIL_Loss_Pain_CE_painLSTM_wln_512_1_seqlen_10_milcepain_weighted_2min_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'

	# meta_dir = '../output/pain_lstm_wln_MIL_Loss_Pain_CE_painLSTM_wln_512_1_seqlen_10_milcepain_weighted_5sec_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_10'

	# meta_dir = '../output/pain_lstm_wbn_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_2min_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'

	# meta_dir = '../output/pain_lstm_wbn_wstep_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_step_5_milcepain_weighted_2min_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'

	# meta_dir = '../output/pain_lstm_wbn_withrot_MIL_Loss_Pain_CE_painLSTM_withrot_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'

	# meta_dir = '../output/pain_lstm_wbn_wapp_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_wapp_milcepain_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'

	# meta_dir = '../output/pain_lstm_wbn_wstep_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_step_5_milcepain_weighted_2min_minsize30_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'

	# meta_dir = '../output/pain_lstm_wbn_512_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_weighted_2min_bw10_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'


	# meta_dir = '../output/pain_lstm_wbn_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_weighted_2min_withval_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'
	# has_val = True

	# meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_Pain_CE_painLSTM_allout_512_1_seqlen_10_milcepain_weighted_2min_withval_bw10_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'
	# meta_dir = '../output/pain_lstm_wbn_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_weighted_2min_withval_bw10_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'
	# has_val = True

	# meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_Pain_CE_painLSTM_allout_512_1_seqlen_10_milcepain_weighted_10min_withval_bw10_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	# has_val = True
	# idx_metric = 5
	# dirs_select = [6]

	# meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_Pain_CE_painLSTM_allout_512_1_seqlen_10_milcepain_weighted_2min_withval_bw10_withRotFlowCropPercentBetterBgOptFlow/LPS_2fps_crop_timeseg_nth_1_nfps_240'
	# has_val = True
	# idx_metric = 5
	# dirs_select = [6]

	# meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_Pain_CE_painLSTM_allout_512_1_seqlen_10_milcepain_weighted_2min_withval_bw10_withRotFlowCropPercentBetterBgOptFlowLW10/LPS_2fps_crop_timeseg_nth_1_nfps_240'
	# has_val = True
	# idx_metric = 5
	
	meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_Pain_CE_painLSTM_allout_512_1_seqlen_10_milcepain_weighted_2min_withval_bw10_epoch20_withRotFlowCropPercentBetterBgOptFlowLW10/LPS_2fps_crop_timeseg_nth_1_nfps_240'
	has_val = True
	idx_metric = 5




	# meta_dir = '../output/pain_lstm_wbn_512_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_weighted_2min_deno8_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'
	# meta_dir = '../output/pain_lstm_wbn_wapp_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_wapp_milcepain_weighted_2min_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'

	# meta_dir = '../output/pain_lstm_wbn_512_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_weighted_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	
	# meta_dir ='../output/pain_lstm_wbn_512_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'


	# meta_dir ='../output/pain_lstm_wbn_512_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_weighted_2min_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_240'

	# meta_dir ='../output/pain_lstm_wbn_512_MIL_Loss_Pain_CE_painLSTM_512_1_seqlen_10_milcepain_weighted_5sec_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_10'
	# meta_dir = '../output/pain_lstm_wbn_MIL_Loss_CE_painLSTM_1024_1_seqlen_10_milce_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200/'
	# meta_dir = '../output/pain_lstm_wbn_MIL_Loss_Mix_painLSTM_1024_1_seqlen_10_latest_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'


	dirs = [val for val in glob.glob(os.path.join(meta_dir, '*')) if os.path.isdir(val)]
	dirs.sort()
	# print (dirs,len(dirs))
	dirs_select
	dirs = [dirs[idx] for idx in dirs_select]
	# if len(dirs)>8:
	# 	dirs = dirs[:8]

	# dirs = dirs[:-1]
	
	# return

	# keys = []
	str_title = '  ,best,end,#best,#end'
	if has_val:
		str_title = '  ,best,end,val,#best,#end,#val'

	accus = {}
	for dir_curr in dirs:
		# print (dir_curr)
		test_file = os.path.join(dir_curr, 'debug_log_testing.txt')
		val_file = os.path.join(dir_curr, 'debug_log_validation.txt')

		test_sub = os.path.split(dir_curr)[-1]
		test_sub = test_sub[test_sub.index('test')+5:test_sub.index('test')+7]
		if os.path.exists(test_file):
			if has_val and os.path.exists(val_file):
				print (val_file)
				_, idx_end_val, _, idx_max_val = parse_test_file(val_file,idx_metric)
				end_accu, idx_end, max_accu, idx_max, val_accu, idx_val = parse_test_file(test_file,idx_metric, idx_max_val)
				str_print = '%.2f,%.2f,%.2f,%d,%d,%d'%(max_accu,end_accu,val_accu,idx_max,idx_end,idx_val)
			else:
				end_accu, idx_end, max_accu, idx_max = parse_test_file(test_file,idx_metric)
				# end_accu = end_accu*100
				# max_accu = max_accu*100
				str_print = '%.2f,%.2f,%d,%d'%(max_accu,end_accu,idx_max,idx_end)

			accus[test_sub] = str_print

	keys = list(accus.keys())
	keys.sort()
	keys = keys[::-1]
	print (meta_dir)
	# str_print = '  ,best,end,#best,#end'
	print (str_title)
	for k in keys:
		print (k+','+accus[k])



if __name__=='__main__':
	main()