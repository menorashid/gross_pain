import torch
import torch.nn as nn
from . import pain_lstm_wbn

class PainHead(pain_lstm_wbn.PainHead):

    def __init__(self, base_network , output_types, n_hidden_to_pain = 1, d_hidden = 512, dropout = 0.5 , seq_len = 10):
        # self.seq_len = seq_len

        # self.encoder = base_network.encoder
        # self.to_3d = base_network.to_3d
        # if n_hidden_to_pain>1:
        #     self.lstm = nn.LSTM(input_size = base_network.dimension_3d , hidden_size = d_hidden, num_layers = n_hidden_to_pain, dropout = dropout)
        # else:
        #     self.lstm = nn.LSTM(input_size = base_network.dimension_3d , hidden_size = d_hidden, num_layers = n_hidden_to_pain)
        super(PainHead, self).__init__(base_network, output_types, n_hidden_to_pain, d_hidden, dropout, seq_len)

        self.to_pain = nn.ModuleList([self.lstm,nn.Sequential(torch.nn.BatchNorm1d(d_hidden, affine=True),nn.Dropout(),nn.Linear(d_hidden,1),nn.Tanh())])

        
        # self.output_types = output_types

    def forward_pain(self, input_dict):
        output_dict = super(PainHead,self).forward_pain(input_dict)
        pain = output_dict['pain']
        # print (pain[:10])
        pain = (pain+1)/2
        # print (pain[:10])
        output_dict['pain'] = pain
        return output_dict