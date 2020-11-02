import torch
import numpy as np
from helpers import util
import imageio
import math
def scratch():
    # seq_len = self.seq_len
    # step_size = self.step_size
    angles = np.linspace(0,2*np.pi,15)
    angle = np.random.choice(angles)
    print (angles, angle)
    rot_mat = util.rotationMatrixXZY(0, angle,0)
    rot_mat = torch.from_numpy(rot_mat).float().cuda()
    print (rot_mat)
    return



    seq_len = 10
    step_size = 5
    
    latent_3d = torch.randn(1156,1024).cuda()
    print (latent_3d.size())
    
    input_size = latent_3d.size(0)
    rem = 1 if (input_size%seq_len)>0 else 0
    batch_size = input_size//seq_len + rem

    rem = seq_len*batch_size - input_size
    padding = torch.zeros((rem,latent_3d.size(1))).cuda()
    latent_3d = torch.cat((latent_3d,padding), axis = 0)
    # print (len(latent_3d))
    latent_3d = [latent_3d[start:start+seq_len] for start in range(0,len(latent_3d)-seq_len+1,step_size)]
    # print (len(latent_3d))
    # print (len(latent_3d[-1]))

    assert (len(latent_3d[-1])==seq_len)

    batch_size = len(latent_3d)


    # if len(latent_3d[-1])<seq_len:
    #     latent_3d = latent_3d[:-1]
    # print (len(latent_3d))
    # print (len(latent_3d[-1]))

    latent_3d = torch.cat(latent_3d,axis = 0)



    # for start in range(0,len(latent_3d),step_size):


    # print (latent_3d.size())

    latent_3d = torch.reshape(latent_3d,(batch_size,seq_len,latent_3d.size(1)))
    latent_3d = torch.transpose(latent_3d, 0,1)
    print (latent_3d.size())
    return


    print (latent_3d.size())
    seq_len = 5
    batch_size = latent_3d.size(0)
    rem = seq_len -1
    padding = torch.zeros((rem,latent_3d.size(1)))
    latent_3d = torch.cat((latent_3d,padding), axis = 0)

    print (latent_3d.size())
    conv = torch.nn.Conv1d(1024, 2048, seq_len)
    latent_3d = latent_3d.unsqueeze(0).transpose(1,2)
    print (latent_3d.size())
    out = conv(latent_3d)
    print (out.size())

    return
    input_size = latent_3d.size(0)
    rem = 1 if (input_size%seq_len)>0 else 0
    batch_size = input_size//seq_len + rem

    rem = seq_len*batch_size - input_size
    padding = torch.zeros((rem,latent_3d.size(1)))
    latent_3d = torch.cat((latent_3d,padding), axis = 0)
    
    latent_3d = torch.reshape(latent_3d,(batch_size,seq_len,latent_3d.size(1)))
    latent_3d = torch.transpose(latent_3d, 0,1)
    
    rnn = torch.nn.LSTM(1024, 2048, 1)
    output, (hn, cn) = rnn(latent_3d)
    print (output.size(),hn.size(),cn.size())
    print (output[4,:3,:2])
    print (hn[:,-1,:2])


def main():
    # img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
    file_name = '../data/pain_no_pain_x2h_intervals_for_extraction_672_380_10fps_oft_0.7_crop/aslan/20190103122542_130850/0_opt/as_00_0_000434.png'
    im = np.array(imageio.imread(file_name)).astype(float)
    print (type(im))
    # .astype(float)
    # im = im/255.
    # mag = im[:,:,2]




    # mag = im[:3,:3,0]
    # mag = im[:3,:3,1]
    # print (np.min(mag), np.max(mag))
    for s in range(3):
        print (im[:3,:3,s])

    # mag_max = math.sqrt(15**2+15**2)
    # print (mag_max)
    # mag_ac = np.sqrt(np.sum(np.square(im[:,:,:2]*30), axis = 2))
    # mag_other = im[:,:,2]*mag_max

    # print (mag_ac.shape, np.min(mag_ac), np.max(mag_ac))
    # print (np.min(mag_other), np.max(mag_other))
    # diff = np.abs(mag_ac - mag_other)
    # print (np.min(diff), np.max(diff), np.mean(diff))



    # mag_ac = np.sqrt(np.sum(np.square(im[:,:,1:]*30), axis = 2))
    # mag_other = im[:,:,0]
    # print (mag_ac.shape, np.min(mag_ac), np.max(mag_ac))
    # print (np.min(im[:,:,1:]), np.max(im[:,:,1:]))
    # diff = np.abs(mag_ac - mag_other)
    # print (np.min(diff), np.max(diff), np.mean(diff))


    # for i in range(im.shape[2]):
    #     print (np.min(im[:,:,i]), np.max(im[:,:,i]))

    # im = torch.from_numpy(imageio.imread(file_name)).float()
    # print (im.type())
    # print (im.size())
    # print (im.size(), torch.min(im), torch.max(im))

        #         img = img.div(255)
                

        # return np.array(self.transform_in(imageio.imread(name)), dtype='float32')


if __name__=='__main__':
    main()