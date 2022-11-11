from cProfile import label
import pandas as pd
import SimpleITK as sitk
import os
import numpy as np
from image_loader import ImageLoadPipe
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#maxi:[500,500,82]
ct_path = 'C:/Users/yyc13/gallbladder/data/training_data(resampled)/ct'
mask_path = 'C:/Users/yyc13/gallbladder/data/training_data(resampled)/gb_label'

train_label_dic =pd.read_excel('X:/dataset/gallbladder/train_dic.xls')
test_label_dic = 'X:/dataset/gallbladder/test_dic.xls'

# files = os.listdir(ct_path)
# print(train_label_dic.values[:,1:])


datapipe = ImageLoadPipe(scan_path = ct_path,
                                        scan_list = pd.read_excel('X:/dataset/gallbladder/train_dic.xls').values[:,1:],
                                        mask_path = mask_path,
                                        if_transform = False,
                                        augmentation_list=['rotate','GaussianBlur','EqualHist'],
                                        if_pad=True,
                                        pad_size=[500,500,82],
                                        if_return_mask=False)
gb_loader = DataLoader(datapipe,batch_size = 1, shuffle = True)
print(len(gb_loader))
img, label = next(iter(gb_loader))
print(img.shape)
print(label) 
print(label.shape)
# plt.imshow(img[0,0,:,:,44],cmap='gray')
# plt.show()
 