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
        
    def get_pred_thresh(self, y_pred, segment_key_pred):
        y_pred = self.collate(y_pred, segment_key_pred, self.deno)
        # .squeeze(dim = 1)

        if y_pred.size(1)==2:
            y_pred = y_pred[:,1]
        else:
            assert y_pred.size(1)==1
            y_pred.squeeze(dim = 1)

        y_pred[y_pred<self.thresh] = 0
        y_pred[y_pred>=self.thresh] = 1
        return y_pred

    def get_pred_argmax(self, y_pred, segment_key_pred, pain = False):
        if pain:
            y_pred = self.collate_pain(y_pred, segment_key_pred, self.deno)
        else:    
            y_pred = self.collate(y_pred, segment_key_pred, self.deno)
        assert y_pred.size(1)==2

        y_pred = torch.argmax(y_pred, dim = 1).type(y_pred.type())
        
        return y_pred

    def get_accuracy(self, y_pred, segment_key_pred, y):

        if self.accuracy=='argmax':
            y_pred = self.get_pred_argmax(y_pred, segment_key_pred)
            loss = (y_pred, y)
        elif self.accuracy=='argmax_pain':
            y_pred = self.get_pred_argmax(y_pred, segment_key_pred, pain = True)
            loss = (y_pred, y)
        elif self.accuracy =='old':
            y_pred = self.get_pred_thresh(y_pred, segment_key_pred)
            loss = torch.eq(y_pred.type(y.type()), y).view(-1)
            loss = torch.sum(loss)/float(loss.size(0))
        elif self.accuracy:
            y_pred = self.get_pred_thresh(y_pred, segment_key_pred)
            loss = (y_pred, y)
        return loss

    def collate_pain(self, y, segment_key, deno):
        assert y.size(1)==2
        
        y_collate = []
        vals, inds = torch.unique_consecutive(input = segment_key, return_inverse = True)
        
        for idx_val, val in enumerate(vals):
            x = y[inds==idx_val,:]
            
            if deno=='random':
                deno_curr = 2**np.random.randint(0,4)
                k = max(1,x.size(0)//deno_curr)
            else:
                k = max(1,x.size(0)//deno)

            _,idx_pmf = torch.sort(x, dim=0, descending=True)
            rel_idx = idx_pmf[:k,1]
            pmf = x[rel_idx,:]
            pmf = torch.sum(pmf[:k,:], dim = 0, keepdims = True)/k
            # print ('pmf',pmf.size())
            y_collate.append(pmf)

        y_collate = torch.cat(y_collate, dim = 0)
        # print ('y_collate.size()',y_collate.size())
        return y_collate
        

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
        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)

        if self.accuracy:
            loss = self.get_accuracy(y_pred, segment_key_pred, y)
        else:
            y_pred = self.collate(y_pred, segment_key_pred, self.deno).squeeze(dim = 1)
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
    def __init__(self, label_key, key_idx, deno, accuracy = False, weights = None):
        super(MIL_Loss_CE, self).__init__(label_key, key_idx, deno, accuracy)
        self.smax = torch.nn.Softmax(dim = 1)
        # print (weights)
        self.loss = torch.nn.CrossEntropyLoss(weight = weights)
        # print (self.loss)
0        # s = input()

    def forward(self, pred_dict, label_dict):
        segment_key = label_dict[self.key_idx]
        y_pred = pred_dict[self.label_key]
        y = label_dict[self.label_key]
        assert y_pred.size(1)==2

        if segment_key.size(0)>y_pred.size(0):
            segment_key_pred = pred_dict[self.key_idx]
        else:
            segment_key_pred = segment_key

        y_pred = self.smax(y_pred)
        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)

        if self.accuracy:
            loss = self.get_accuracy(y_pred, segment_key_pred, y)
        else:
            y_pred = self.collate(y_pred, segment_key_pred, self.deno)
            loss = self.loss(y_pred, y)
        return loss

class MIL_Loss_Pain_CE(MIL_Loss_CE):

    def forward(self, pred_dict, label_dict):
        segment_key = label_dict[self.key_idx]
        y_pred = pred_dict[self.label_key]
        y = label_dict[self.label_key]
        assert y_pred.size(1)==2

        if segment_key.size(0)>y_pred.size(0):
            segment_key_pred = pred_dict[self.key_idx]
        else:
            segment_key_pred = segment_key

        y_pred = self.smax(y_pred)
        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)

        if self.accuracy:
            loss = self.get_accuracy(y_pred, segment_key_pred, y)
        else:
            y_pred = self.collate_pain(y_pred, segment_key_pred, self.deno)
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
        

        y_pred = self.smax(y_pred)
        
        y = y.type(y_pred.type())                
        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)
        
        if self.accuracy:
            loss = self.get_accuracy(y_pred, segment_key_pred, y)
        else:
            y_pred = self.collate(y_pred, segment_key_pred, self.deno)[:,1]
            loss = self.loss(y_pred, y)


        return loss


class MIL_Loss_justGT(MIL_Loss):
    def __init__(self, label_key, key_idx, deno, accuracy = False):
        super(MIL_Loss_justGT, self).__init__(label_key, key_idx, deno, accuracy)
        
        self.smax = torch.nn.Softmax(dim = 1)        
        # self.loss = torch.nn.BCELoss()
        self.thresh = 0.5
        self.loss = None


    def forward(self, pred_dict, label_dict):
        segment_key = label_dict[self.key_idx]
        y_pred = pred_dict[self.label_key]
        y = label_dict[self.label_key]

        if segment_key.size(0)>y_pred.size(0):
            segment_key_pred = pred_dict[self.key_idx]
        # print ('hey')
        else:
            segment_key_pred = segment_key

        bef_type = y.type()
        # print (type(y))
        # print (torch.nn.functional.one_hot(y))
        y = y.type(y_pred.type())        
        y = self.collate(y.view((y.size(0),1)), segment_key, 1).squeeze(dim = 1)
            
        if self.accuracy:
            loss = self.get_accuracy(y_pred, segment_key_pred, y)
        else:

            y_pred = self.smax(y_pred)
            y_pred = self.collate(y_pred, segment_key_pred, self.deno).squeeze(dim = 1)
            y_pred = self.smax(y_pred)
            # print ('y_pred',y_pred)
            # print ('y',y)

            y = y.type(bef_type)
            # print ('y',y)
            y = torch.nn.functional.one_hot(y)
            # print ('y',y)

            y_pred = torch.log(y_pred)
            # print ('y_pred',y_pred)
            loss = -1*y*y_pred
            # print ('loss',loss)
            loss = torch.mean(torch.sum(loss,dim = 1))
            # print ('loss',loss)
            # s = input()

        return loss

    def get_accuracy(self,y_pred, segment_key_pred, y):
        check = self.smax(y_pred)
        check = self.collate(check, segment_key_pred, 8).squeeze(dim = 1)
        print (check,y)
            
        y_pred = torch.argmax(y_pred, dim = 1).type(y_pred.type())
        # print (y_pred.size(),y_pred[:10])
        y_pred = self.collate(y_pred.view((y_pred.size(0),1)), segment_key_pred, 1).squeeze(dim = 1)
        # print (y_pred)
        y_pred[y_pred<self.thresh] = 0
        y_pred[y_pred>=self.thresh] = 1
        # print (y_pred)
        # print (y)
        # print ('y_pred.size(), y.size()',y_pred.size(), y.size())
        loss = torch.eq(y_pred, y).view(-1)
        # print ('y_pred.size(), y.size()',y_pred.size(), y.size())
        loss = torch.sum(loss)/float(loss.size(0))

        return loss



