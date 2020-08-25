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
	config_file_model = 'configs/config_train_rotation_crop_newCal.py'
	job_name_model = 'withRotCropNewCal'
	epoch = '50'


	# config_file = 'configs/config_train_painfromlatent_crop.py'
	# job_name = 'painWithRotCropwApp'
	# python_path = 'train_encode_decode_pain_wApp.py'

	config_file = 'configs/config_train_painfromlatent_crop.py'
	job_name = 'pain'
	python_path = 'train_encode_decode_pain.py'

	data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
	
	util.mkdir('to_runs')

	num_gpus = 1
	num_per_gpu = 4
	for idx in range(num_gpus):
		test_horses = test_horses_all[num_per_gpu*idx:num_per_gpu*idx+num_per_gpu]
		out_file = os.path.join('to_runs','_'.join(['to_run',to_run_str,job_name,str(idx)]))
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
			break
		break
		# print (commands)
		util.writeFile(out_file, commands)


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
	pnp_latent_commands()
	# back_bone_commands()
	# debug_commands()



if __name__=='__main__':
	main()

