from helpers import util
import os

train_horses = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
test_horses_all = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
# ['herrera','julia','naughty_but_nice']
# ['inkasso', 'kastanjett', 'sir_holger']
# 
config_file = 'configs/config_train_rotation_crop.py'
data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
job_name = 'withRotCrop'
util.mkdir('to_runs')

num_gpus = 1
num_per_gpu = 8
for idx in range(num_gpus):
	test_horses = test_horses_all[num_per_gpu*idx:num_per_gpu*idx+num_per_gpu]
	out_file = os.path.join('to_runs','to_run_1_'+job_name+'_'+str(idx))
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