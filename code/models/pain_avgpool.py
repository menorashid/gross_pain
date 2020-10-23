import torch
import torch.nn as nn
raw_input = input

class PainHead(nn.Module):
    def __init__(self, base_network , output_types, n_hidden_to_pain = 2, d_hidden = 2048, dropout = 0.5 , seq_len = 9):
        super(PainHead,self).__init__()
        self.seq_len = seq_len
        self.encoder = base_network.encoder
        self.to_3d = base_network.to_3d

        assert n_hidden_to_pain == 2

        lin_parts = []
        d_in = base_network.dimension_3d
        d_out = d_hidden
        for i in range(n_hidden_to_pain):
            lin_part = []
            lin_part.append(nn.Linear(d_in, d_out))
            lin_part.append(nn.ReLU())
            lin_part.append(nn.BatchNorm1d(d_out, affine=True))
            if dropout>0:
                lin_part.append(nn.Dropout(dropout))
            # lin_part = nn.Sequential(*lin_part)
            lin_parts.append(lin_part)
            d_in = d_out
            # d_out = d_hidden

        pool = nn.AvgPool1d(seq_len, stride = 1, padding = seq_len//2, count_include_pad = False)
        pain = nn.Linear(d_hidden, 2)
        pain = lin_parts[1]+[pain]
        pain = nn.Sequential(*pain)
        bef_pool = nn.Sequential(*lin_parts[0])

        self.to_pain = nn.ModuleList([bef_pool,pool,pain])
        self.output_types = output_types
    

    def forward_pain(self, input_dict):
        
        input = input_dict['img_crop']
        segment_key = input_dict['segment_key']
        batch_size = input_dict['img_crop'].size()[0]
        device = input.device
        
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat)
        
        latent_3d = self.to_pain[0](latent_3d)

        latent_pooled_all = []
        for segment_val in torch.unique_consecutive(segment_key):
            segment_bin = segment_key==segment_val
            latent_pooled = latent_3d[segment_bin,:]
            
            init_size = latent_pooled.size(0)
            latent_pooled = latent_pooled.unsqueeze(0).transpose(1,2)
            latent_pooled = self.to_pain[1](latent_pooled)
            latent_pooled = latent_pooled.squeeze(0).transpose(0,1)
            
            assert latent_pooled.size(0)==init_size
            latent_pooled_all.append(latent_pooled)

        latent_pooled_all = torch.cat(latent_pooled_all, axis = 0)

        output_pain = self.to_pain[2](latent_pooled_all)
        pain_pred = torch.nn.functional.softmax(output_pain, dim = 1)

        ###############################################
        # Select the right output
        output_dict_all = {'pain': output_pain, 'pain_pred': pain_pred, 'segment_key':segment_key} 
        output_dict = {}
        # print (self.output_types)
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict 