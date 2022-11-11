from torch import classes
import torch
from train_manager import TrainManager
from ConvModel import ConvModel
from image_loader import ImageLoadPipe
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from loss_func import DiceLoss
import pandas as pd


class TrainBuilder():
    def __init__(self,paths, kwargs):
        self.tf_path = paths['tfsummary']
        self.result_savepath = paths['result']
        self.model_SavePath = paths['model']
        self.traindata_loader = DataLoader(ImageLoadPipe(scan_path=paths['train_scan'],
                                                scan_list=pd.read_excel(paths['train_dic']).values[:,1:],
                                                mask_path=paths['train_mask'],
                                                if_transform=False,
                                                augmentation_list=None,
                                                if_pad=True,
                                                pad_size=[500,500,82],
                                                if_return_mask=False),
                                        batch_size=1,shuffle=True)
        self.valdata_loader = DataLoader(ImageLoadPipe(scan_path=paths['val_scan'],
                                                scan_list=pd.read_excel(paths['val_dic']).values[:,1:],
                                                mask_path=paths['val_mask'],
                                                if_transform=False,
                                                augmentation_list=None,
                                                if_pad=True,
                                                pad_size=[500,500,82],
                                                if_return_mask=False),
                                        batch_size=1,shuffle=True)
        self.kwargs = kwargs
        self.epoch = kwargs['epoch']
        self.device = kwargs['device']
        self.learning_rate = kwargs['lr']
        self.model = ConvModel(img_channel = 1, classes = 2, elu = True)
        self.loss_func = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1,2]))
        self.loss_func.to(self.device)
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),lr=self.learning_rate,
                                    momentum=0.9)
        if paths['loadmodel']!=None:
            print('loading trained model')
            checkpoint = torch.load(paths['loadmodel'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.train()

        self.TrainManager = TrainManager()

    def run_train(self):
        self.TrainManager.begin_run(self.kwargs, self.model,self.traindata_loader,self.tf_path,self.device)
        for e_ind in range(self.epoch):
            self.TrainManager.begin_epoch()
            i = 0
            for data in self.traindata_loader:
                img,label = data
                self.optimizer.zero_grad()
                img = img.to(self.device)
                label = label.to(self.device)
                prediction = self.model(img)
                loss = self.loss_func(prediction,label)  #runtime error: 0D or 1D tensor supported
                loss.backward()
                self.optimizer.step()
                self.TrainManager.track_step(loss,label,prediction,i)
                i+=1
                del img, label
                torch.cuda.empty_cache()
            self.model.train(False)
            self.TrainManager.begin_validation()
            val_i = 0
            for val_data in self.valdata_loader:
                val_img,val_label = val_data
                self.TrainManager.sum_img(val_img,val_i,'img_validation')
                val_img = val_img.to(self.device)
                val_label = val_label.to(self.device)
                val_prediction = self.model(val_img)
                val_loss = self.loss_func(val_prediction,val_label)
                self.TrainManager.track_val_step(val_loss, val_label,val_prediction)
                val_i+=1
            self.TrainManager.end_validation(len(self.valdata_loader))
            self.TrainManager.end_epoch()
            self.TrainManager.save_best_model(self.model,self.optimizer,self.model_SavePath)
            self.model.train(True)
            self.TrainManager.save_result_csv(self.result_savepath,'train_result')
        self.TrainManager.end_run()


path_local = {'train_scan':'C:/Users/yyc13/gallbladder/data/training_data(resampled)/ct',
              'train_mask':'C:/Users/yyc13/gallbladder/data/training_data(resampled)/gb_label',
              'val_scan':'C:/Users/yyc13/gallbladder/data/training_data(resampled)/ct',
              'val_mask':'C:/Users/yyc13/gallbladder/data/training_data(resampled)/gb_label',

              'tfsummary':'C:/Users/yyc13/gallbladder/DL_result/tfsummary',
              'result':'C:/Users/yyc13/gallbladder/DL_result/result',
              'model':'C:/Users/yyc13/gallbladder/DL_result/model',
              'loadmodel':None,

              'train_dic':'X:/dataset/gallbladder/train_dic.xls',
              'val_dic':'X:/dataset/gallbladder/test_dic.xls'}

path_cluster= {'train_scan':'/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
              'train_mask':'/data/p288821/dataset/gallbladder/training_data(resampled)/gb_label',
              'val_scan':'/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
              'val_mask':'/data/p288821/dataset/gallbladder/training_data(resampled)/gb_label',

              'tfsummary':'/data/p288821/tfsummary/gallbladder',
              'result':'/data/p288821/result/gallbladder',
              'model':'/data/p288821/model/gallbladder',
              'loadmodel':None,#'/data/p288821/gallbladder_project/epoch_25.pt',

              'train_dic':'/data/p288821/dataset/gallbladder/train_dic.xls',
              'val_dic':'/data/p288821/dataset/gallbladder/test_dic.xls'}

path_gnl_cluster = {'train_scan':'/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
              'train_mask':'/data/p288821/dataset/gallbladder/training_data(resampled)/GnL_label',
              'val_scan':'/data/p288821/dataset/gallbladder/training_data(resampled)/ct',
              'val_mask':'/data/p288821/dataset/gallbladder/training_data(resampled)/GnL_label',

              'tfsummary':'/data/p288821/tfsummary/gallbladder',
              'result':'/data/p288821/result/gallbladder',
              'model':'/data/p288821/model/gallbladder',
              'loadmodel':'/data/p288821/gallbladder_project/GnL_result/epoch_33.pt',

              'train_dic':'/data/p288821/dataset/gallbladder/train_dic.xls',
              'val_dic':'/data/p288821/dataset/gallbladder/test_dic.xls'}
              
params = {'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          'lr':0.001,
          'epoch':100}

TrainBuilder(paths = path_cluster, kwargs=params).run_train()

