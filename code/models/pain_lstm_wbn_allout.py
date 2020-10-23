import torch
import torch.nn as nn
from . import pain_lstm_wbn

raw_input = input

class PainHead(pain_lstm_wbn.PainHead):

    def forward_pain(self, input_dict):
        input = input_dict['img_crop']
        segment_key = input_dict['segment_key']
        batch_size = input_dict['img_crop'].size()[0]
        device = input.device
        
        out_enc_conv = self.encoder(input)
        center_flat = out_enc_conv.view(batch_size,-1)
        latent_3d = self.to_3d(center_flat)
        
        h_n_all = []
        segment_key_new = []
        for segment_val in torch.unique_consecutive(segment_key):
            segment_bin = segment_key==segment_val
            rel_latent = latent_3d[segment_bin,:]
            
            init_size = rel_latent.size(0)

            rel_latent = self.pad_input(rel_latent)
            out,_ = self.lstm(rel_latent)
            
            out = out.view(out.size(0)*out.size(1),-1)
            
            out = out[:init_size,:]
                        
            h_n_all.append(out)
            # segment_key_rel = segment_key[segment_bin][:h_n.size(0)]
            # segment_key_new.append(segment_key_rel)

        h_n = torch.cat(h_n_all, axis = 0)
        output_pain = self.to_pain[1](h_n)
        # segment_key_new = torch.cat(segment_key_new, axis=0)
        
        pain_pred = torch.nn.functional.softmax(output_pain, dim = 1)

        ###############################################
        # Select the right output
        output_dict_all = {'pain': output_pain, 'pain_pred': pain_pred, 'segment_key':segment_key} 
        output_dict = {}
        # print (self.output_types)
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict    

