import torch.nn as nn
import torchvision


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2,
                 which_cnn='inceptionv3',
                 pretrained=True,
                 input_size=128,
                 output_types='pain'
                 ):
        super(CNNClassifier, self).__init__()
        self.num_classes = num_classes
        self.which_cnn = which_cnn
        self.pretrained = pretrained
        self.input_size = input_size
        self.output_types = output_types

        if self.which_cnn == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True, progress=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        if self.which_cnn == 'inception_v3':
            # Same settings as "Dynamics are Important..."
            extra_fc_units = 512
            # Global avg. pool is already applied before fc in pytorch implementation
            self.model = torchvision.models.inception_v3(pretrained=pretrained,
                                                         progress=True,
                                                         aux_logits=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, extra_fc_units)
            self.fc_out = nn.Linear(extra_fc_units, self.num_classes)


    def _set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):

        if self.which_cnn == 'inception_v3':  # Pad to 299x299
            x = nn.functional.pad(input=x, pad=(85,86,85,86))
        x = self.model(x)
        y_pred = self.fc_out(x)

        return y_pred
        
        # ###############################################
        # # Select the right output
        # output_dict_all = {'pain' : y_pred}
        # output_dict = {}
        # for key in self.output_types:
        #     output_dict[key] = output_dict_all[key]

        # return output_dict

