from torch import classes
import torch
from ConvModel import ConvModel
from image_loader import ImageLoadPipe
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd

paths_gnl = {'train_scan': '/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
                    'train_mask': '/data/p288821/dataset/gallbladder/training_data(resampled)/GnL_label',
                    'val_scan': '/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
                    'val_mask': '/data/p288821/dataset/gallbladder/training_data(resampled)/GnL_label',

                    'tfsummary': '/data/p288821/tfsummary/gallbladder',
                    'result': '/data/p288821/result/gallbladder',
                    'model': '/data/p288821/model/gallbladder',
                    'loadmodel': '/data/p288821/gallbladder_project/GnL_result(train_R2)/epoch_17.pt',

                    'train_dic': '/data/p288821/dataset/gallbladder/train_dic.xls',
                    'val_dic': '/data/p288821/dataset/gallbladder/test_dic.xls'}

paths_g = {'train_scan': '/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
            'train_mask':'/data/p288821/dataset/gallbladder/training_data(resampled)/gb_label',
             'val_scan':'/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
              'val_mask':'/data/p288821/dataset/gallbladder/training_data(resampled)/gb_label',

              'tfsummary':'/data/p288821/tfsummary/gallbladder',
              'result':'/data/p288821/result/gallbladder',
              'model':'/data/p288821/model/gallbladder',
              'loadmodel':'/data/p288821/gallbladder_project/G_weighted_tresults(R2)/epoch_35.pt',

              'train_dic':'/data/p288821/dataset/gallbladder/train_dic.xls',
              'val_dic':'/data/p288821/dataset/gallbladder/test_dic.xls'}

valdata_loader = DataLoader(ImageLoadPipe(scan_path=paths_g['val_scan'],
                scan_list=pd.read_excel(paths_g['val_dic']).values[:,1:],
                mask_path=paths_g['val_mask'],
                if_transform=False,
                augmentation_list=None,
                if_pad=True,
                pad_size=[500,500,82],
                if_return_mask=False),
        batch_size=1,shuffle=True)
model = ConvModel(img_channel = 1, classes = 2, elu = True)
if paths_g['loadmodel'] != None:
    print('loading trained model')
    checkpoint = torch.load(paths_g['loadmodel'],map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
loss_func = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1,2]))
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_func.to(device)
model.to(device)

val_i = 0
pre_dict = {}
label_dict = {}
for val_data in valdata_loader:
    val_img,val_label = val_data
    val_img = val_img.to(device)
    val_label = val_label.to(device)
    val_prediction = model(val_img)
    # val_loss = loss_func(val_prediction,val_label)
    print(val_prediction,val_label)

    pre_dict[str(val_i)] = np.squeeze(val_prediction.detach().numpy().tolist())
    label_dict[str(val_i)] = val_label.detach().numpy().tolist()
    val_i+=1
    # print(pre_dict)

pre_dict = pd.DataFrame(pre_dict)
pre_dict = pre_dict.T
pre_dict.to_csv('/data/p288821/gallbladder_project/G_weighted_tresults(R2)/test_prediction(e35).csv')

label_dict = pd.DataFrame(label_dict)
label_dict = label_dict.T
label_dict.to_csv('/data/p288821/gallbladder_project/G_weighted_tresults(R2)/test_label(e35).csv')


