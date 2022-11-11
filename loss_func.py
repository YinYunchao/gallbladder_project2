
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
@Y.Yin 2022
this file is used to supplement loss function selections beyond what pytorch offers
cited: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
'''
class DiceLoss(nn.Module):
    def __init__(self, used_sigmoid_actFun = False):
        super(DiceLoss,self).__init__()
        self.if_sigmoid = used_sigmoid_actFun

    def forward(self, prediction, label, smooth = 1):
        '''
        Dice loss for unbalanced classification
        func: DSC = 2|X ∩ Y| / (|X|+|Y|)
        if the model used sigmoid activation function, then the prediction does not need to 
        be sigmoided again, changed by parameter used_sigmoid_actFun
        '''
        if not self.if_sigmoid:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.view(-1)
        label = label.view(-1)
        intersection = (prediction*label).sum()
        dice_loss = 1 - (2.*intersection+smooth)/(prediction.sum() + label.sum() + smooth)
        return dice_loss


class DiceBCELoss(nn.Module):
    '''
    combine the dice loss and cross entropy loss to diverse the loss for optimization
    '''
    def __init__(self,used_sigmoid_actFun = False):
        super(DiceBCELoss,self).__init__()
        self.if_sigmoid = used_sigmoid_actFun

    def forward(self, prediction, label, smooth = 1):
        if not self.if_sigmoid:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.view(-1)
        label = label.view(-1)
        intersection = (prediction*label).sum()
        dice_loss = 1-(2 * intersection + smooth / (prediction.sum()+label.sum()+smooth))
        BCE = F.binary_cross_entropy(prediction,label,reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

class IoULoss(nn.Module):
    '''
    Intersection over Union (IoU) loss, aka Jaccard loss, is similar to dice but calculated as the ratio
    between the overlap of the positive instances between two sets, and their mutual combined values

    J(X,Y) = |X ∩ Y| / |X ∪ Y| = |X ∩ Y| /(|X|+|Y|-|X ∩ Y|)
    '''
    def __init__(self,used_sigmoid_actFun = False):
        super(IoULoss, self).__init__()
        self.if_sigmoid = used_sigmoid_actFun

    def forward(self,prediction, label, smooth = 1):
        if not self.if_sigmoid:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.view(-1)
        label = label.view(-1)
        intersection = (prediction*label).sum()
        total = (prediction+label).sum()
        union = total-intersection
        IoU_loss = 1 - (intersection+smooth)/(union+smooth)
        return IoU_loss

class FocalLoss(nn.Module):
    '''
    REF: https://arxiv.org/abs/1708.02002
    '''
    def __init__(self,used_sigmoid_actFun = False):
        super(FocalLoss).__init__()
        self.if_sigmoid = used_sigmoid_actFun

    def forward(self, prediction, label, alpha = 0.8, gamma = 2, smooth = 1):
        if not self.if_sigmoid:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.view(-1)
        label = label.view(-1)
        BCE = F.binary_cross_entropy(prediction,label,reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE        
        return focal_loss

class TverskyLoss(nn.Module):
    '''
    REF: https://arxiv.org/abs/1706.05721
    '''
    def __init__(self, used_sigmoid_actFun = False):
        super(TverskyLoss, self).__init__()
        self.if_sigmoid = used_sigmoid_actFun

    def forward(self, prediction, label, alpha = 0.5, beta = 0.5, smooth = 1):
        if not self.if_sigmoid:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.view(-1)
        label = label.view(-1)
        TP = (prediction*label).sum()
        FP = ((1-label) * prediction).sum()
        FN = (label * (1-label)).sum()
        Tversky_loss = 1 - (TP + smooth) / (TP + alpha*FP + beta*FN +smooth) 
        return Tversky_loss

class FocalTverskyLoss(nn.Module):
    '''
    Combination of Tversky loss and focal loss, use the gamma modifier of Focal loss
    '''
    def __init__(self, used_sigmoid_actFun = False):
        super(FocalTverskyLoss, self).__init__()
        self.if_sigmoid = used_sigmoid_actFun

    def forward(self, prediction, label, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        if not self.if_sigmoid:
            prediction = torch.sigmoid(prediction)       
        prediction = prediction.view(-1)
        label = label.view(-1)
        TP = (prediction * label).sum()    
        FP = ((1-label) * prediction).sum()
        FN = (label * (1-prediction)).sum()
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
        return FocalTversky

class ComboLoss(nn.Module):
    '''
    a combination of Dice Loss and a modified Cross-Entropy function
    cited: https://arxiv.org/abs/1805.02798
    '''
    def __init__(self, used_sigmoid_actFun = False):
        super(ComboLoss, self).__init__()
        self.if_sigmoid = used_sigmoid_actFun

    def forward(self, prediction, label, smooth=1, CE_RATIO=0.5, alpha=0.5, beta=0.5, eps=1e-9):
        if not self.if_sigmoid:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.view(-1)
        label = label.view(-1)
        intersection = (prediction * label).sum()    
        dice = (2. * intersection + smooth) / (prediction.sum() + label.sum() + smooth)
        prediction = torch.clamp(prediction, eps, 1.0 - eps)       
        out = - (alpha * ((label * torch.log(prediction)) + ((1 - alpha) * (1.0 - label) * torch.log(1.0 - prediction))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        return combo