import torch.nn as nn
import torch
from torchsummary import summary

def passthrough(img,**kwargs):
    return img

def activation_func(elu, nchan):
    '''
    offer two choices of activation function
    '''
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

def _make_nConv(nchan, depth, elu, kernel_size):
    '''
    this func generate the convolutional layers by loop

    param:
    nchan: input channel number
    depth: number of convolutional layers generated, loop times
    elu: param for activation func choice
    '''
    layers = []
    for _ in range(depth):
        layers.append(conv_comblayer(nchan=nchan, elu=elu,kernel_size=kernel_size))
    return nn.Sequential(*layers)

class conv_comblayer(nn.Module):
    '''
    the combination components of convolutional layers
    including:conv layer, activation func, batch normalization
    '''
    def __init__(self, nchan, elu, kernel_size = 5):
        super(conv_comblayer, self).__init__()
        self.relu1 = activation_func(elu=elu, nchan=nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm3d(nchan)
    def forward(self, img):
        out = self.relu1(self.bn1(self.conv1(img)))
        return out


class Conv_layer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride, nConv, elu=True, if_dropout = False):
        super(Conv_layer, self).__init__()
        self.pool= nn.Conv3d(in_channels=channel_in, out_channels=channel_out,
        kernel_size=2, stride=stride) #this is pooling layer, not extracting features, stride=2
        self.conv_layers = _make_nConv(nchan=channel_out, depth=nConv, elu=elu, kernel_size=kernel_size)
        self.dropout_layer = passthrough
        if if_dropout:
            self.dropout_layer = nn.Dropout3d()

    def forward(self, img):
        out = self.pool(img)
        out = self.dropout_layer(out)
        out = self.conv_layers(out)
        return out
        




class ConvModel(torch.nn.Module):
    def __init__(self,elu,img_channel,classes):
        super(ConvModel, self).__init__()
        self.InputLayer = torch.nn.Conv3d(img_channel,16,5) #(in_channel, out_channel, kernel size)
        self.conv1 = Conv_layer( channel_in = 16, channel_out = 32, kernel_size = 5, 
                                stride = [2,2,1], nConv=1, elu=elu, if_dropout = False)#[250,250,82]
        self.conv2 = Conv_layer( channel_in = 32, channel_out = 32, kernel_size = 5, 
                                stride = [2,2,2], nConv=1, elu=elu, if_dropout = False)#[125,125,41]
        self.conv3 = Conv_layer( channel_in = 32, channel_out = 64, kernel_size = 5, 
                                stride = [2,2,2], nConv=1, elu=elu, if_dropout = False)#[62,62,20]
        self.conv4 = Conv_layer( channel_in = 64, channel_out = 64, kernel_size = 5, 
                                stride = [2,2,2], nConv=1, elu=elu, if_dropout = False)#[31,31,10]
        self.conv5 = Conv_layer( channel_in = 64, channel_out = 128, kernel_size=3,
                                stride = [2,2,2], nConv=2, elu=elu, if_dropout=True)#[15,15,5]
        self.conv6 = Conv_layer( channel_in = 128, channel_out = 128, kernel_size=3,
                                stride = [2,2,2], nConv=1, elu=elu, if_dropout=True)#[7,7,2]
        
        self.fc1 = torch.nn.Linear(12544,120)
        self.fc_relu1 = activation_func(elu=elu, nchan=60)
        self.fc2 = torch.nn.Linear(120,10)
        self.fc_relu2 = activation_func(elu=elu, nchan=10)
        self.fc3 = torch.nn.Linear(10,classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self,img):
        out = self.InputLayer(img)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(-1,12544) # 12544 is pre-calculated by func num_flat_features
        out = self.fc_relu1(self.fc1(out))
        out = self.fc_relu2(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        return out

    def num_flat_features(self,x):
        '''
        calculate the feature map size after flatten
        preused, because input size is the same
        '''
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features*=s
        return num_features

    def test(self, classes, img_channel, device = 'cpu'):
        '''
        to test whether bugs in the model builded above, input is randomly generated
        '''
        input_tensor = torch.rand(1, 1,500,500,82)
        label = torch.rand(1,classes)
        out = self.forward(input_tensor)
        print(out)
        assert label.shape==out.shape
        summary(self.to(torch.device(device=device)),(img_channel,500,500,82),device=device)
        print("ConvModel test is complete")