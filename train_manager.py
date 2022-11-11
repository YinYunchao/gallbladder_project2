import imp
import time
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainManager():
    '''
    The class to refactor the training loop, functions including:
        --count running time of each epoch and validation,
        --generate tf summary
        --summary the first image and label from loader, for chekcing img and mask
        --summary the graph of model
        --save best-performed model on validation set
        '''
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_acc = 0
        self.epoch_start_time = None

        self.val_running_loss = 0
        self.val_running_acc = 0
        self.val_start_time = None
        self.val_data = []
        self.val_loss_record = []
        self.val_acc_record = []

        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tf_writer = None

        self.saved_model_num = 0

    
    def begin_run(self,run,network,loader,tfsummary_path,device):
        '''
        Execute at the begining of training;
        params:
        run: the hyper parameter used in this training run
        network: the model
        loader: the loader
        tfsummary_path: the folder(absolute path) to save tf record 9
        '''
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count+=1
        self.network = network
        self.loader = loader
        self.tf_writer = SummaryWriter(tfsummary_path)

        img,label = next(iter(self.loader))
        img = img.to(device, dtype = torch.float)
        self.tf_writer.add_image(str(np.squeeze(label.detach().numpy())),
                                img[0,0,:,:,int(img.shape[-1]/2)],
                                dataformats='HW')
        self.tf_writer.add_graph(self.network,img)

    def end_run(self):
        '''
        Execute at the end of training;
        '''
        self.tf_writer.flush()
        self.tf_writer.close()
        self.epoch_count = 0
    
    def begin_epoch(self):
        '''
        Execute at the begining of each epoch;
        '''
        self.epoch_start_time = time.time()
        self.epoch_count+=1
        self.epoch_loss = 0
        self.epoch_acc = 0
        # print('epoch_',self.epoch_count)
    
    def end_epoch(self):
        '''
        Execute at the end of each epoch;
        '''
        epoch_duration= time.time()-self.epoch_start_time
        run_duration = time.time()-self.run_start_time

        loss = self.epoch_loss/len(self.loader)
        acc = self.epoch_acc/len(self.loader)
        self.tf_writer.add_scalar('epoch_loss', loss, self.epoch_count)
        self.tf_writer.add_scalar('epoch_acc', acc, self.epoch_count)
        if self.epoch_count%5==0:
            for name, param in self.network.named_parameters():
                self.tf_writer.add_histogram(name, param, self.epoch_count)
                self.tf_writer.add_histogram(f'{name}.grad',param.grad,self.epoch_count)
        results = OrderedDict()
        # results["run"] = self.run_count
        results["epoch_ind"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = acc
        results["epoch_duration"] = epoch_duration
        results["run_duration"] = run_duration
        for key in self.run_params: results[key] = self.run_params[key]
        self.run_data.append(results)

    def track_step(self,loss,label,prediction,step):
        self.epoch_loss+=loss.item()
        self.tf_writer.add_scalar('step_loss', loss.item(), step+(self.epoch_count-1)*len(self.loader))
        self.epoch_acc = self.epoch_acc+ self.acc_cal(label,prediction)
    
    def acc_cal(self, label, prediction):
        label_npy = label.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        if (prediction[0,1]>=0.5 and label_npy[0]==1) or (prediction[0,1]<0.5 and label_npy[0]==0):
            return 1
        else:
            return 0



    def begin_validation(self):
        '''
        Execute at the begining of validation;
        '''
        self.val_running_loss = 0
        self.val_running_acc = 0
        self.val_start_time = time.time()

    def track_val_step(self, val_loss, label, prediction):
        self.val_running_loss+=val_loss.item()
        self.val_running_acc = self.val_running_acc + self.acc_cal(label,prediction)
        
    def end_validation(self,val_dataNum):
        '''
        Execute at the end of validation;
        '''
        val_loss = self.val_running_loss/val_dataNum
        val_acc = self.val_running_acc/val_dataNum
        val_results = OrderedDict()
        val_results["epoch_ind"] = self.epoch_count
        val_results["loss"] = val_loss
        val_results["accuracy"] = val_acc
        val_results["epoch_duration"] = time.time()-self.val_start_time
        #print('time for validation: ', val_results['epoch_duration'], ';',val_results['loss'])
        self.val_loss_record.append(val_loss)
        self.val_acc_record.append(val_acc)
        self.val_data.append(val_results)
        self.val_running_loss = self.val_running_loss/val_dataNum
        self.tf_writer.add_scalar('epoch_validation_loss:',
                                    val_loss,self.epoch_count)
        self.tf_writer.add_scalar('epoch_validation_acc',
                                    val_acc,self.epoch_count)
        
        
    def save_result_csv(self, save_path, fileName):
        '''
        save the running loss, running time etc during the training
        '''
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(os.path.join(save_path,f'{fileName}.csv'))
        pd.DataFrame.from_dict(
            self.val_data, orient='columns'
        ).to_csv(os.path.join(save_path,'val_{}.csv'.format(fileName)))

    def save_best_model(self, model, optimizer, save_path):
        '''
        The model will be saved if the validation loss is the smallest among all epochs
        '''
        if self.epoch_count>2 and self.val_running_loss <= min(self.val_loss_record[:-1]):
            saved_models = sorted(os.listdir(save_path))
            if len(saved_models)>5:
                for file in saved_models[:-2]:
                    os.remove(os.path.join(save_path,file))
            torch.save({'epoch':self.epoch_count,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},
                       os.path.join(save_path,'epoch_{}.pt'.format(self.epoch_count)))


    def load_saved_model(self,load_path):
        model = torch.load(load_path)
        model.eval()

    def sum_img(self,img, i, name):
        if i==0:
            self.tf_writer.add_image(name,
            img[0,0,:,:,int(img.shape[-1]/2)],dataformats='HW',
            global_step=self.epoch_count)




    

