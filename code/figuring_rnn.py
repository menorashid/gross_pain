import torch

def main():
	latent_3d = torch.randn(1159,1024)
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





# 	rnn = nn.LSTM(10, 20, 2)
# >>> input = torch.randn(5, 3, 10)
# >>> h0 = torch.randn(2, 3, 20)
# >>> c0 = torch.randn(2, 3, 20)
# >>> output, (hn, cn) = rnn(input, (h0, c0))

# input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.

# h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.

# c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.

# If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.


if __name__=='__main__':
	main()