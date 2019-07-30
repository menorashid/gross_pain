from test import *
from helpers import util, visualize

def main():
	vid_arr = ['ch01_20181209092600'  ,'ch03_20181209092844'  ,'ch05_20181208230458'  ,'ch06_20181209093714']

	data_dir = '../data/tester_kit/naughty_but_nice'
	out_dir = os.path.join('../scratch','test_first_frame')
	util.mkdir(out_dir)

	model_path = '../data/deeplab_xception_coco_voctrainval.tar.gz'
	model = DeepLabModel(model_path)

	frames = [(val*20)+1 for val in range(5)]
	frames = ['frame_'+'%09d'%val+'.jpg' for val in frames]
	
	for vid_curr in vid_arr:

		for frame_curr in frames:
			frame_curr = os.path.join(data_dir, vid_curr, frame_curr)
			out_dir_anno = os.path.join(data_dir, vid_curr+'_anno')
			util.mkdir(out_dir_anno)
			out_file = os.path.join(out_dir_anno, os.path.split(frame_curr)[1].replace('.jpg','.png'))
			run_visualization(model, frame_curr, out_file)
			print out_file

	visualize.writeHTMLForFolder(out_dir)


if __name__=='__main__':
	main()