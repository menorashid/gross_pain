import numpy as np

from skimage import io
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, rescale
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import os
from multiview_frame_extractor import MultiViewFrameExtractor
from helpers import util, visualize

def save_frames():
	meta_dir = '../scratch/matching_im'
	util.mkdir(meta_dir)
	data_path = '../data/lps_data/surveillance_camera'
	mfe = MultiViewFrameExtractor(data_path = data_path)

	subject = 'sir_holger'
	time_str = '20181102111500'
	views = [0,1,2,3]
	out_dir = os.path.join(meta_dir, subject)
	util.mkdir(out_dir)

	out_files = mfe.extract_single_time( subject, time_str, views, out_dir)
	visualize.writeHTMLForFolder(out_dir)

	
def main():
	# save_frames()

	meta_dir = '../scratch/matching_im'
	in_dir = os.path.join(meta_dir, 'sir_holger')
	im_format = 'si_%d_20181102111500.jpg'
	for views in [[0,1],[1,2],[2,3],[3,0]]:
		ims = [os.path.join(in_dir, im_format%view) for view in views]
		out_file = os.path.join(in_dir, 'correspondences_%d_%d.jpg'%(views[0],views[1]))


		img_left = rescale(io.imread(ims[0]),scale = 0.25).squeeze()
		img_right = rescale(io.imread(ims[1]),scale = 0.25).squeeze()
		
		# Find sparse feature correspondences between left and right image.

		descriptor_extractor = ORB()

		descriptor_extractor.detect_and_extract(img_left)
		keypoints_left = descriptor_extractor.keypoints
		descriptors_left = descriptor_extractor.descriptors

		descriptor_extractor.detect_and_extract(img_right)
		keypoints_right = descriptor_extractor.keypoints
		descriptors_right = descriptor_extractor.descriptors

		matches = match_descriptors(descriptors_left, descriptors_right,
		                            cross_check=True)

		# Estimate the epipolar geometry between the left and right image.

		model, inliers = ransac((keypoints_left[matches[:, 0]],
		                         keypoints_right[matches[:, 1]]),
		                        FundamentalMatrixTransform, min_samples=8,
		                        residual_threshold=1, max_trials=5000)

		inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
		inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

		print(f"Number of matches: {matches.shape[0]}")
		print(f"Number of inliers: {inliers.sum()}")
		
		plt.figure()
		
		plt.gray()

		plot_matches(plt.gca(), img_left, img_right, keypoints_left, keypoints_right,
		             matches[inliers], only_matches=True)
		plt.gca().axis("off")
		plt.gca().set_title("Inlier correspondences")

		plt.savefig(out_file)
		plt.close()
		print (out_file)

	visualize.writeHTMLForFolder(in_dir)

	

if __name__=='__main__':
	main()