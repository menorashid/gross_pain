
from models.pain_rotAllCat import PainHead as PainRotAllCat

class PainHead(PainRotAllCat):
    def __init__(self, base_network , output_types):
        super(PainHead, self).__init__(base_network, output_types, n_hidden_to_pain = 3, d_hidden = 1024, num_views = 4)