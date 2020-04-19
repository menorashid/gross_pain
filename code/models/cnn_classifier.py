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
        if self.which_cnn == 'inceptionv3':
            self.model = torchvision.models.inception_v3(pretrained=True, progress=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)


    def _set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        x = input_dict['img_crop']
        y_pred = self.model(x)
        
        ###############################################
        # Select the right output
        output_dict_all = {'pain' : y_pred}
        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict

