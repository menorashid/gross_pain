import torch
import numpy as np
from .mil_losses import *

class LossOnDict(torch.nn.Module):
    def __init__(self, key, loss):
        super(LossOnDict, self).__init__()
        self.key = key
        self.loss = loss
        
    def forward(self, pred_dict, label_dict):
        loss = self.loss(pred_dict[self.key], label_dict[self.key])
        return loss

class LossOnPredDict(torch.nn.Module):
    def __init__(self, key, other_key, loss, weight = 1):
        super(LossOnPredDict, self).__init__()
        self.key = key
        self.other_key = other_key
        self.loss = loss
        self.weight = weight
        
    def forward(self, pred_dict, label_dict):
        loss = self.weight * self.loss(pred_dict[self.key], pred_dict[self.other_key])
        return loss

class LossWeightedOnDict(torch.nn.Module):
    def __init__(self, key, weight_key, loss_weight = 1):
        super(LossWeightedOnDict, self).__init__()
        self.loss = torch.nn.MSELoss(reduction = 'none')
        self.key = key
        self.weight_key = weight_key
        self.weight = loss_weight

    def forward(self, pred_dict, label_dict):
        loss = self.loss(pred_dict[self.key], label_dict[self.key])
        # print (pred_dict.keys(), label_dict.keys())
        opt_flow = label_dict[self.weight_key]
        opt_flow = opt_flow.unsqueeze(1)
        # print (torch.min(loss), torch.max(loss))
        # print (torch.min(opt_flow), torch.max(opt_flow))
        loss = opt_flow*loss
        loss = self.weight * torch.mean(loss)
        # print (loss)
        return loss
        # print (opt_flow.size())
        # print ('loss.size()',loss.size())
        # s = input()



class PreApplyCriterionListDict(torch.nn.Module):
    """
    Wraps a loss operating on tensors into one that processes dict of labels and predictions
    """
    def __init__(self, criterions_single, sum_losses=True, loss_weights=None):
        super(PreApplyCriterionListDict, self).__init__()
        self.criterions_single = criterions_single
        self.sum_losses = sum_losses
        self.loss_weights = loss_weights

    def forward(self, pred_dict, label_dict):
        """
        The loss is computed as the sum of all the loss values
        :param pred_dict: List containing the predictions
        :param label_dict: List containing the labels
        :return: The sum of all the loss values computed
        """
        losslist = []
        for criterion_idx, criterion_single in enumerate(self.criterions_single):
            loss_i = criterion_single(pred_dict, label_dict)
            
            # print (criterion_idx, loss_i)

            if self.loss_weights is not None:
                loss_i = loss_i * self.loss_weights[criterion_idx]
            losslist.append(loss_i)

        if self.sum_losses:
            return sum(losslist)
        else:
            return losslist    


class LossLabel(torch.nn.Module):
    def __init__(self, key, loss_single):
        super(LossLabel, self).__init__()
        self.key = key
        self.loss_single = loss_single

    def forward(self, preds, labels):
        pred_pose = preds[self.key]
        label_pose = labels[self.key]

        return self.loss_single.forward(pred_pose, label_pose)
     
class LossLabelMeanStdNormalized(torch.nn.Module):
    """
    Normalize the label before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, subjects=False, weight=1):
        super(LossLabelMeanStdNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.subjects = subjects
        self.weight=weight

    def forward(self, preds, labels):
        pred_pose = preds[self.key]
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        label_pose_norm = (label_pose-label_mean)/label_std

        if self.subjects:
            info = labels['frame_info']
            subject = info.data.cpu()[:,3]
            errors = [self.loss_single.forward(pred_pose[i], label_pose_norm[i]) for i,x in enumerate(pred_pose) if subject[i] in self.subjects]
            #print('subject',subject,'errors',errors)
            if len(errors) == 0:
                return torch.autograd.Variable(torch.FloatTensor([0])).cuda()
            return self.weight * sum(errors) / len(errors)

        return self.weight * self.loss_single.forward(pred_pose,label_pose_norm)
    
class LossLabelMeanStdUnNormalized(torch.nn.Module):
    """
    UnNormalize the prediction before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, scale_normalized=False, weight=1):
        super(LossLabelMeanStdUnNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.scale_normalized = scale_normalized
        #self.subjects = subjects
        self.weight=weight

    def forward(self, preds, labels):
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        pred_pose = preds[self.key]
        
        if self.scale_normalized:
            per_frame_norm_label = label_pose.norm(dim=1).expand_as(label_pose)
            per_frame_norm_pred  = pred_pose.norm(dim=1).expand_as(label_pose)
            pred_pose = pred_pose / per_frame_norm_pred * per_frame_norm_label

        pred_pose_norm = (pred_pose*label_std) + label_mean
        
        return self.weight*self.loss_single.forward(pred_pose_norm, label_pose)
