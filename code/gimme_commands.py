from helpers import util
import os

def back_bone_commands():
	# train_horses = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# test_horses_all = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# # ['herrera','julia','naughty_but_nice']
	# # ['inkasso', 'kastanjett', 'sir_holger']
	# # 
	# config_file = 'configs/config_train_rotation_newCal.py'
	# data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
	# # data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	# job_name = 'withRotNewCal'
	# util.mkdir('to_runs')

	train_horses = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	test_horses_all = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# ['herrera','julia','naughty_but_nice']
	# ['inkasso', 'kastanjett', 'sir_holger']
	# 
	config_file = 'configs/config_rotcrop_debug.py'
	# data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	job_name = 'withRotCropDebug'

	util.mkdir('to_runs')


	num_gpus = 2
	num_per_gpu = 4
	for idx in range(num_gpus):
		test_horses = test_horses_all[num_per_gpu*idx:num_per_gpu*idx+num_per_gpu]
		out_file = os.path.join('to_runs','to_run_'+job_name+'_'+str(idx))
		print (out_file)
	# test_horses = ['brava', 'herrera']
	# test_horses = ['inkasso','julia']
	# test_horses = ['kastanjett', 'naughty_but_nice']
	# test_horses = ['sir_holger']
	# 
	# 
	# 
	# , 'inkasso','julia']
	# , 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# test_horses = ['aslan', 'brava', 'herrera', 'inkasso']

		commands = []
		for test_subject in test_horses:
			train_subjects = [x for x in train_horses if x is not test_subject]
			str_com = ['python','train_encode_decode.py']
			str_com+= ['--config_file', config_file]
			str_com+= ['--dataset_path', data_path]
			str_com+= ['--train_subjects', '/'.join(train_subjects)]
			str_com+= ['--test_subjects', test_subject]
			str_com+= ['--job_identifier', job_name]
			str_com = ' '.join(str_com)
			commands.append(str_com)
			print (str_com)

		# print (commands)
		util.writeFile(out_file, commands)


def pnp_latent_commands():
	train_horses = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	test_horses_all = train_horses[:]
	# , 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# ['herrera','julia','naughty_but_nice']
	# ['inkasso', 'kastanjett', 'sir_holger']

	
	to_run_str = 'nth1'
	# 
	# config_file_model = 'configs/config_train_rotation_crop_newCal.py'
	# job_name_model = 'withRotCropNewCal'

	# config_file_model = 'configs/config_train_rotFlowCrop.py'
	# job_name_model = 'withRotFlowCropPercent'

	# config_file_model = 'configs/config_train_rotFlowCropLatent.py'
	# job_name_model = 'withRotFlowCropLatentPercentLatentLr0.1'

	config_file_model = 'configs/config_train_rotFlowCropBetterBg.py'
	job_name_model = 'withRotFlowCropPercentBetterBg'


	epoch = '50'


	# config_file = 'configs/config_train_painfromlatent_crop.py'
	# job_name = 'painWithRotCropwApp'
	# python_path = 'train_encode_decode_pain_wApp.py'

	# config_file = 'configs/config_train_painfromlatent_crop.py'
	# config_file = 'configs_pain/config_train_painBN_crop_timeseg_random_llr.py'
	# job_name = 'painDenoRandomLLR'
	
	config_file = 'configs_pain/config_train_painRot2world.py'
	job_name = 'painRot2World'

	config_file = 'configs_pain/config_train_painRotAllCat_3fc_1024.py'
	job_name = 'painRotAllCat_3fc_1024'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_milce.py'
	job_name = 'painLSTM_1024_1_seqlen_10_milce'
	# job_name = 'painLSTM_exp'
	config_file = 'configs_pain/config_train_pain_lstm_wbn_binary.py'
	job_name = 'painLSTM_1024_1_seqlen_10_wbn_binary'
	
	config_file = 'configs_pain/config_train_pain_avgpool.py'
	job_name = 'painAvgPool'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_512.py'
	job_name = 'painLSTM_512_1_seqlen_10'
	
	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milce.py'
	job_name = 'painLSTM_512_1_seqlen_10_milce'


	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_weighted.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_weighted'

	config_file = 'configs_pain/config_train_pain_lstm_wln_512_milcepain_weighted_5sec.py'
	job_name = 'painLSTM_wln_512_1_seqlen_10_milcepain_weighted_5sec'

	config_file = 'configs_pain/config_train_pain_lstm_wln_512_milcepain_weighted_2min.py'
	job_name = 'painLSTM_wln_512_1_seqlen_10_milcepain_weighted_2min'


	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_weighted_2min.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_weighted_2min'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_2min.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_2min'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_wstep_512_milcepain_weighted_2min.py'
	job_name = 'painLSTM_512_1_seqlen_10_step_5_milcepain_weighted_2min'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_wstep_512_milcepain_weighted_2min_minsize30.py'
	job_name = 'painLSTM_512_1_seqlen_10_step_5_milcepain_weighted_2min_minsize30'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_withrot.py'
	job_name = 'painLSTM_withrot'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_wapp_512_milcepain_weighted_2min.py'
	job_name = 'painLSTM_512_1_seqlen_10_wapp_milcepain_weighted_2min'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_weighted_2min_bw10.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_weighted_2min_bw10'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_weighted_2min_deno8.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_weighted_2min_deno8'

	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_weighted_2min_llr.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_weighted_2min'

	# config_file = 'configs_pain/config_train_pain_lstm_wbn_wapp_512_milcepain.py'
	# job_name = 'painLSTM_512_1_seqlen_10_wapp_milcepain'

	# config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milce_fix.py'
	# job_name = 'painLSTM_512_1_seqlen_10_debug'
	

	# config_file = 'configs_pain/config_train_pain_lstm_wbn_512_2.py'
	# job_name = 'painLSTM_512_2_seqlen_10'
		
	# config_file = 'configs_pain/config_train_pain_lstm_wbn_allout_512.py'
	# job_name = 'painLSTM_512_1_seqlen_10_wbn_allout'
	
	# config_file = 'configs_pain/config_train_pain_fully_lstm_wbn.py'
	# job_name = 'painFullyLSTM_512_1_seqlen_10_wbn_milce'

	# config_file = 'configs_pain/config_train_pain_fully_lstm_wbn_2layer.py'
	# job_name = 'painFullyLSTM_512_2_seqlen_10_wbn_milce'

	# config_file = 'configs_pain/config_train_pain_fully_lstm_wbn_allout.py'
	# job_name = 'painFullyLSTM_512_1_seqlen_10_wbn_allout_milce'

	# config_file = 'configs_pain/config_train_pain_fully_lstm_wbn_allout_2layer.py'
	# job_name = 'painFullyLSTM_512_2_seqlen_10_wbn_allout_milce'
	
	# config_file = 'configs_pain/config_train_pain_lstm_wbnrelu.py'
	# job_name = 'painLSTM_2048_2_seqlen_10'
	# config_file = 'configs_pain/config_train_pain_conv1d.py'
	# job_name = 'painConv1d_exp'

	python_path = 'train_encode_decode_pain.py'
	# data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    
	
	util.mkdir('to_runs')

	num_gpus = 1
	num_per_gpu = 8
	for idx in range(num_gpus):
		test_horses = test_horses_all[num_per_gpu*idx:num_per_gpu*idx+num_per_gpu]
		out_file = os.path.join('to_runs','_'.join(['to_run',to_run_str,job_name,job_name_model,str(idx)]))
		print (out_file)

		commands = []
		for test_subject in test_horses:
			train_subjects = [x for x in train_horses if x is not test_subject]
			str_com = ['python',python_path]
			str_com+= ['--config_file', config_file]
			str_com+= ['--dataset_path', data_path]
			str_com+= ['--train_subjects', '/'.join(train_subjects)]
			str_com+= ['--test_subjects', test_subject]
			str_com+= ['--job_identifier', job_name]

			str_com+= ['--config_file_model', config_file_model]
			str_com+= ['--train_subjects_model', '/'.join(train_subjects)]
			str_com+= ['--test_subjects_model', test_subject]
			str_com+= ['--job_identifier_encdec', job_name_model]
			str_com+= ['--epoch_encdec', epoch]

			str_com = ' '.join(str_com)
			commands.append(str_com)
			print (str_com)
		# 	break
		# break
		# print (commands)
		util.writeFile(out_file, commands)
		print (out_file)


def pnp_latent_commands_withval():
	train_horses = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	val_horses = ['naughty_but_nice','aslan']
	test_horses_all = train_horses[:]
	# , 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# ['herrera','julia','naughty_but_nice']
	# ['inkasso', 'kastanjett', 'sir_holger']

	
	to_run_str = 'nth1'
	# 
	# config_file_model = 'configs/config_train_rotation_crop_newCal.py'
	# job_name_model = 'withRotCropNewCal'

	# config_file_model = 'configs/config_train_rotFlowCrop.py'
	# job_name_model = 'withRotFlowCropPercent'

	# config_file_model = 'configs/config_train_rotFlowCropLatent.py'
	# job_name_model = 'withRotFlowCropLatentPercentLatentLr0.1'

	config_file_model = 'configs/config_train_rotFlowCropBetterBg.py'
	job_name_model = 'withRotFlowCropPercentBetterBg'

	# config_file_model = 'configs/config_train_rotFlowCropBetterBgOptFlow.py'
	# job_name_model = 'withRotFlowCropPercentBetterBgOptFlow'
	

	epoch = '50'


	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_weighted_2min_withval.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_weighted_2min_withval'
	
	config_file = 'configs_pain/config_train_pain_lstm_wbn_512_milcepain_weighted_2min_withval_bw10.py'
	job_name = 'painLSTM_512_1_seqlen_10_milcepain_weighted_2min_withval_bw10'
	
	config_file = 'configs_pain/config_train_pain_lstm_wbn_allout_512_milcepain_weighted_2min_withval_bw10.py'
	job_name = 'painLSTM_allout_512_1_seqlen_10_milcepain_weighted_2min_withval_bw10'
	
	# config_file = 'configs_pain/config_train_pain_lstm_wbn_allout_512_milcepain_weighted_10min_withval_bw10.py'
	# job_name = 'painLSTM_allout_512_1_seqlen_10_milcepain_weighted_10min_withval_bw10'
	
	config_file = 'configs_pain/config_train_pain_lstm_wbn_allout_512_milcepain_weighted_5min_withval_bw10.py'
	job_name = 'painLSTM_allout_512_1_seqlen_10_milcepain_weighted_5min_withval_bw10'
			

	python_path = 'train_encode_decode_pain.py'
	# data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop'
    
	
	util.mkdir('to_runs')

	num_gpus = 2
	num_per_gpu = 4
	for idx in range(num_gpus):
		test_horses = test_horses_all[num_per_gpu*idx:num_per_gpu*idx+num_per_gpu]
		out_file = os.path.join('to_runs','_'.join(['to_run',to_run_str,job_name,job_name_model,str(idx)]))
		print (out_file)

		commands = []
		for test_subject in test_horses:
			val_subject = val_horses[0] if test_subject is not val_horses[0] else val_horses[1]

			train_subjects = [x for x in train_horses if (x is not test_subject) and (x is not val_subject)]
			train_subjects_model = [x for x in train_horses if x is not test_subject]
			str_com = ['python',python_path]
			str_com+= ['--config_file', config_file]
			str_com+= ['--dataset_path', data_path]
			str_com+= ['--train_subjects', '/'.join(train_subjects)]
			str_com+= ['--test_subjects', test_subject]
			str_com+= ['--val_subjects', val_subject]
			str_com+= ['--job_identifier', job_name]

			str_com+= ['--config_file_model', config_file_model]
			str_com+= ['--train_subjects_model', '/'.join(train_subjects_model)]
			str_com+= ['--test_subjects_model', test_subject]
			str_com+= ['--job_identifier_encdec', job_name_model]
			str_com+= ['--epoch_encdec', epoch]

			str_com = ' '.join(str_com)
			commands.append(str_com)
			print (str_com)
		# 	break
		# break
		# print (commands)
		util.writeFile(out_file, commands)
		print (out_file)

def debug_commands():
	train_horses = ['aslan' , 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	test_horses_all = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# ['herrera','julia','naughty_but_nice']
	# ['inkasso', 'kastanjett', 'sir_holger']
	# 
	config_file = 'configs/config_train_rotation_translation_newCal.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
	# data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	job_name = 'withRotTransAll'
	
	config_file = 'configs/config_train_rotCrop_segmask.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	job_name = 'withRotCropSeg'

	config_file = 'configs/config_train_rot_segmask.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
	job_name = 'withRotSeg'

	# config_file = 'configs/config_train_rotTranslate_segmask.py'
	# data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
	# job_name = 'withRotTranslateSeg'

	config_file = 'configs/config_train_rotation_newCal.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_128_128_2fps/'
	job_name = 'withRotNewCal'

	config_file = 'configs/config_train_rotation_crop_newCal.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	job_name = 'withRotCropNewCal'

	config_file = 'configs/config_train_rotCropLatent.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	job_name = 'withRotCropLatent'

	config_file = 'configs/config_train_rotFlowCropLatent.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/'
	job_name = 'withRotFlowCropLatentPercentLatentLr0.1'

	config_file = 'configs/config_train_rotFlowCropBetterBg.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/'
	job_name = 'withRotFlowCropPercentBetterBg'

	config_file = 'configs/config_train_rotFlowCrop.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/'
	job_name = 'withRotFlowCropPercent'

	config_file = 'configs/config_train_rotFlowCropBetterBgOptFlow.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/'
	job_name = 'withRotFlowCropPercentBetterBgOptFlow'

	config_file = 'configs/config_train_rotFlowCropBetterBgOptFlow_lw10.py'
	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/'
	job_name = 'withRotFlowCropPercentBetterBgOptFlowLW10'

	# config_file = 'configs/config_train_rotCropSegMaskLatent.py'
	# data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	# job_name = 'withRotCropSegLatent'

	util.mkdir('to_runs')

	num_gpus = 2
	num_per_gpu = 4
	for idx in range(num_gpus):
		test_horses = test_horses_all[num_per_gpu*idx:num_per_gpu*idx+num_per_gpu]
		out_file = os.path.join('to_runs','to_run_'+job_name+'_'+str(idx))
		print (out_file)
		commands = []
		for test_subject in test_horses:
			train_subjects = [x for x in train_horses if x is not test_subject]
			# train_subjects = train_horses
			str_com = ['python','train_encode_decode.py']
			str_com+= ['--config_file', config_file]
			str_com+= ['--dataset_path', data_path]
			str_com+= ['--train_subjects', '/'.join(train_subjects)]
			str_com+= ['--test_subjects', test_subject]
			str_com+= ['--job_identifier', job_name]
			str_com = ' '.join(str_com)
			print (str_com)
			commands.append(str_com)
		# 	break
		# break
		util.writeFile(out_file, commands)

def main():
	# pnp_latent_commands()
	pnp_latent_commands_withval()
	# back_bone_commands()
	# debug_commands()
	# out_file = 'get_counts'
	# strs = []
	# train_horses = ['aslan' , 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
	# for horse_id in train_horses:
	# 	str_curr = 'echo {}\ncat ../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/{}_reduced_2fps_frame_index_withSegIndexAndKey.csv | grep "{},0" | wc -l\ncat ../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/{}_reduced_2fps_frame_index_withSegIndexAndKey.csv | grep "{},1" | wc -l'.format(horse_id,horse_id,horse_id,horse_id,horse_id)
	# 	strs.append(str_curr)
	# util.writeFile(out_file,strs)
	# print ('sh '+out_file)





if __name__=='__main__':
	main()

