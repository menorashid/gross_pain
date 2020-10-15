from . import pain_lstm_wbn
import torch
import torch.nn as nn

class PainHead(pain_lstm_wbn.PainHead):
    def __init__(self, base_network , output_types, n_hidden_to_pain = 1, d_hidden = 1024, dropout = 0.5 , seq_len = 10):
        super(PainHead, self).__init__(base_network, output_types, n_hidden_to_pain, d_hidden, dropout, seq_len)
        
        # print (self.to_pain)
        pain_models = list(self.to_pain[-1].children())
        pain_models = pain_models[:-1]+[nn.Linear(d_hidden, 1), nn.Sigmoid()]
        pain_models = nn.Sequential(*pain_models)
        self.to_pain[-1] = pain_models
        # print (self.to_pain)
        # print ('hello')
        # s = input()

        # self.to_pain = nn.ModuleList([self.lstm,nn.Sequential(nn.Dropout(dropout),nn.Linear(d_hidden,2))])

        # MLP.MLP_fromLatent(d_in = base_network.dimension_3d, d_hidden=d_hidden, d_out=2, n_hidden=n_hidden_to_pain, dropout=dropout)
        
        # self.output_types = output_types