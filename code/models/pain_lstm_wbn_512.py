
from . import pain_lstm_wbn

class PainHead(pain_lstm_wbn.PainHead):

    def __init__(self, base_network , output_types, n_hidden_to_pain = 1, d_hidden = 512, dropout = 0.5 , seq_len = 10):
        super(PainHead, self).__init__(base_network, output_types, n_hidden_to_pain, d_hidden, dropout, seq_len)
        