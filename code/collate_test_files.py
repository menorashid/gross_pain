from helpers import util
import glob
import os
import numpy as np
def parse_test_file(test_file):
	lines = util.readLinesFromFile(test_file)[1:]
	accus = np.array([float(line.split(',')[-1]) for line in lines])
	idx_max = np.argmax(accus)
	end_accu = accus[-1]
	max_accu = accus[idx_max]
	return end_accu, len(accus), max_accu, idx_max+1

def main():
	meta_dir = '../output/pain_lstm_wbn_512_MIL_Loss_CE_painLSTM_512_1_seqlen_10_milce_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	# meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_Mix_painLSTM_1024_1_seqlen_10_wbn_allout_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	meta_dir = '../output/pain_lstm_wbn_allout_MIL_Loss_CE_painLSTM_1024_1_seqlen_10_wbn_allout_milce_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'
	
	# meta_dir = '../output/pain_lstm_wbn_MIL_Loss_CE_painLSTM_1024_1_seqlen_10_milce_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200/'
	# meta_dir = '../output/pain_lstm_wbn_MIL_Loss_Mix_painLSTM_1024_1_seqlen_10_latest_withRotFlowCropPercentBetterBg/LPS_2fps_crop_timeseg_nth_1_nfps_1200'

	dirs = [val for val in glob.glob(os.path.join(meta_dir, '*')) if os.path.isdir(val)]
	# keys = []
	accus = {}
	for dir_curr in dirs:
		test_file = os.path.join(dir_curr,'debug_log_testing.txt')

		test_sub = os.path.split(dir_curr)[-1]
		# print (test_sub)
		test_sub = test_sub[test_sub.index('test')+5:test_sub.index('test')+7]
		# print (test_sub)
		if os.path.exists(test_file):
			# keys.append(test_sub)
			end_accu, idx_end, max_accu, idx_max = parse_test_file(test_file)
			end_accu = end_accu*100
			max_accu = max_accu*100
			str_print = '%.2f,%.2f,%d,%d'%(max_accu,end_accu,idx_max,idx_end)
			accus[test_sub] = str_print
	keys = list(accus.keys())
	keys.sort()
	keys = keys[::-1]
	print (meta_dir)
	str_print = '  ,best,end,#best,#end'
	print (str_print)
	for k in keys:
		print (k+','+accus[k])



if __name__=='__main__':
	main()