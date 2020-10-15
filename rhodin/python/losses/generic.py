import torch
import numpy as np

class MIL_Loss(torch.nn.Module):
    def __init__(self, label_key, key_idx, deno, accuracy = False):
        super(MIL_Loss, self).__init__()
        self.label_key = label_key
        self.key_idx = key_idx
        self.deno = deno
        
        # bce
        self.loss = torch.nn.BCELoss()
        self.thresh = 0.5

        self.accuracy = accuracy
        

    def forward(self, pred_dict, label_dict):
        segment_key = label_dict[self.key_idx]
        y_pred = pred_dict[self.label_key]
        y = label_dict[self.label_key]

        if segment_key.size(0)>y_pred.size(0):
            segment_key_pred = pred_dict[self.key_idx]
        # print ('hey')
        else:
            segment_key_pred = segment_key

        y = y.type(y_pred.type())        
        y_pred = self.collate(y_pred, segment_key_pred, self.deno).squeeze(dim = 1)

        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)

        if self.accuracy:
            y_pred[y_pred<self.thresh] = 0
            y_pred[y_pred>=self.thresh] = 1
            loss = torch.eq(y_pred.type(y.type()), y).view(-1)
            loss = torch.sum(loss)/float(loss.size(0))
        else:
            loss = self.loss(y_pred, y)

        return loss


    def collate(self, y, segment_key, deno):
        y_collate = []
        vals, inds = torch.unique_consecutive(input = segment_key, return_inverse = True)
        # print (y.size())
        for idx_val, val in enumerate(vals):
            x = y[inds==idx_val,:]
            
            if deno=='random':
                deno_curr = 2**np.random.randint(0,4)
                k = max(1,x.size(0)//deno_curr)
            else:
                k = max(1,x.size(0)//deno)

            pmf,_ = torch.sort(x, dim=0, descending=True)
            pmf = pmf[:k,:]
            pmf = torch.sum(pmf[:k,:], dim = 0, keepdims = True)/k
            # print ('pmf',pmf.size())
            y_collate.append(pmf)

        y_collate = torch.cat(y_collate, dim = 0)
        # print ('y_collate.size()',y_collate.size())
        return y_collate


class MIL_Loss_CE(MIL_Loss):
    def __init__(self, label_key, key_idx, deno, accuracy = False):
        super(MIL_Loss_CE, self).__init__(label_key, key_idx, deno, accuracy)
        self.smax = torch.nn.Softmax(dim = 1)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred_dict, label_dict):
        segment_key = label_dict[self.key_idx]
        y_pred = pred_dict[self.label_key]
        y = label_dict[self.label_key]
        assert y_pred.size(1)==2

        if segment_key.size(0)>y_pred.size(0):
            segment_key_pred = pred_dict[self.key_idx]
        # print ('hey')
        else:
            segment_key_pred = segment_key

        y_pred = self.smax(y_pred)
        y_pred = self.collate(y_pred, segment_key_pred, self.deno)

        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)

        if self.accuracy:
            y_pred = y_pred[:,1]
            y_pred[y_pred<self.thresh] = 0
            y_pred[y_pred>=self.thresh] = 1
            loss = torch.eq(y_pred.type(y.type()), y).view(-1)
            loss = torch.sum(loss)/float(loss.size(0))
        else:
            loss = self.loss(y_pred, y)
        return loss


class MIL_Loss_Mix(MIL_Loss):
    def __init__(self, label_key, key_idx, deno, accuracy = False):
        super(MIL_Loss_Mix, self).__init__(label_key= label_key, key_idx = key_idx, deno = deno, accuracy = accuracy)
        self.smax = torch.nn.Softmax(dim = 1)
        

    def forward(self, pred_dict, label_dict):
        segment_key = label_dict[self.key_idx]
        y_pred = pred_dict[self.label_key]
        assert y_pred.size(1)==2

        if segment_key.size(0)>y_pred.size(0):
            segment_key_pred = pred_dict[self.key_idx]
        # print ('hey')
        else:
            segment_key_pred = segment_key

        y = label_dict[self.label_key]
        

        y = y.type(y_pred.type())        
        y_pred = self.smax(y_pred)
        # print (y_pred.size(), y.size(), segment_key.size(),pred_dict[self.key_idx].size())
        y_pred = self.collate(y_pred, segment_key_pred, self.deno)[:,1]
        # print ('hey')

        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)
        # print ('hey y')

        # print (y_pred.size(),y.size())
        if self.accuracy:
            # y_pred = y_pred[:,1]
            # .squeeze(dim = 1)
            # print (y_pred, y)
            y_pred[y_pred<self.thresh] = 0
            y_pred[y_pred>=self.thresh] = 1
            loss = torch.eq(y_pred.type(y.type()), y).view(-1)
            loss = torch.sum(loss)/float(loss.size(0))
        else:
            loss = self.loss(y_pred, y)

        return loss


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
