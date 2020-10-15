from . import pain_lstm_wbn
import torch
import torch.nn as nn

class PainHead(pain_lstm_wbn.PainHead):
    def __init__(self, base_network , output_types, n_hidden_to_pain = 2, d_hidden = 2048, dropout = 0.5 , seq_len = 10):
        
        self.seq_len = seq_len
        self.encoder = base_network.encoder
        self.to_3d = base_network.to_3d

        conv_part = []
        d_in = base_network.dimension_3d
        d_out = d_hidden
        for i in range(n_hidden_to_pain):
            conv_part.append(ConvSeg(d_in, d_hidden, seq_len, relu=True, bnorm = True, dropout = dropout))
            d_in = d_hidden

        # conv_part = nn.Sequential(*conv_part)
        self.to_pain = nn.ModuleList(conv_part+[nn.Linear(d_hidden,2)])

        
        self.output_types = output_types