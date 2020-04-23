train_horses = ['aslan', 'brava', 'herrera', 'inkasso', 'julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
test_horses = ['julia', 'kastanjett', 'naughty_but_nice', 'sir_holger']
# test_horses = ['aslan', 'brava', 'herrera', 'inkasso']

config_file = 'configs/config_train_rotation_crop.py'
data_path = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_0.2fps_crop/'
job_name = 'withRotCrop'

for test_subject in test_horses:
	train_subjects = [x for x in train_horses if x is not test_subject]
	str_com = ['python','train_encode_decode.py']
	str_com+= ['--config_file', config_file]
	str_com+= ['--dataset_path', data_path]
	str_com+= ['--train_subjects', '/'.join(train_subjects)]
	str_com+= ['--test_subjects', test_subject]
	str_com+= ['--job_identifier', job_name]
	str_com = ' '.join(str_com)
	print (str_com)