row = 255*np.array([0.485, 0.456, 0.406])
row = row[np.newaxis,np.newaxis,:]
im = np.ones((128,128,3))*row
out_file = '../data/blank_mean.jpg'
im = im.astype(np.uint8)
print (im[0,0,:])
print (im.shape)
imageio.imwrite(out_file, im)
im_l = imageio.imread(out_file)
print (im[0,0,:])
print (im.shape)
