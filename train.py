import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from math import sin, cos
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils
import gc

import utils
import dataset 
import pretreat
import model

def criterion(prediction, mask, regr,weight=0.4, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
  
    # Sum
    loss = weight*mask_loss +(1-weight)* regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss ,mask_loss , regr_loss

def train(epoch, history=None):
    model.train()
    iteration = 0
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(train_loader):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        optimizer.zero_grad()
        output = model(img_batch)
        if epoch < SWITCH_LOSS_EPOCH :
            loss,mask_loss, regr_loss = criterion(output, mask_batch, regr_batch,1)
        else:
            loss,mask_loss, regr_loss = criterion(output, mask_batch, regr_batch,0.5)  
        
        if iteration % 10 == 0:
            print('epoch:{} iteration:{} loss:{:.3f} mask_loss:{:.3f} regr_loss:{:.3f}'.format(epoch+1, iteration+1, loss.data, mask_loss.data, regr_loss.data))
        
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()
        
        optimizer.step()
        exp_lr_scheduler.step()
        iteration += 1

    
    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}\tMaskLoss: {:.6f}\tRegLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data,
        mask_loss.data,
        regr_loss.data))

def evaluate(epoch, history=None):
    model.eval()
    loss = 0
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            if epoch < SWITCH_LOSS_EPOCH :
                loss,mask_loss, regr_loss= criterion(output, mask_batch, regr_batch,1, size_average=False)
                valid_loss += loss.data
                valid_mask_loss += mask_loss.data
                valid_regr_loss += regr_loss.data
            else :
                loss,mask_loss, regr_loss = criterion(output, mask_batch, regr_batch,0.5, size_average=False)
                valid_loss += loss.data
                valid_mask_loss += mask_loss.data
                valid_regr_loss += regr_loss.data 

    
    valid_loss /= len(dev_loader.dataset)
    valid_mask_loss /= len(dev_loader.dataset)
    valid_regr_loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = valid_loss.cpu().numpy()
        history.loc[epoch, 'mask_loss'] = valid_mask_loss.cpu().numpy()
        history.loc[epoch, 'regr_loss'] = valid_regr_loss.cpu().numpy()
    
    torch.save(model.state_dict(), 'model/resnet152/epoch{}.pth'.format(epoch+1))
    
    print('Dev loss: {:.4f}'.format(valid_loss))
    
if __name__ == '__main__':
    SWITCH_LOSS_EPOCH = 5
    
    PATH = '/apl/kaggle/baidu_drive/data/'
 
    train_images_dir = os.path.join(PATH, 'image/train/{}.jpg')
    test_images_dir = os.path.join(PATH, 'image/test/{}.jpg')
    
    train_csv = pd.read_csv(os.path.join(PATH, 'train.csv'))
    test_csv = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
    
    camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    
    df_train, df_dev = train_test_split(train_csv, test_size=0.01, random_state=42)
    df_test = test_csv
        
    train_dataset = dataset.CarDataset(df_train, train_images_dir)
    dev_dataset = dataset.CarDataset(df_dev, train_images_dir)
    test_dataset = dataset.CarDataset(df_test, test_images_dir)
    
    BATCH_SIZE = 1
    
    # Create data generators - they will produce batches
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    
    # Gets the GPU if there is one, otherwise the cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    n_epochs = 15 #6

    model = model.CentResnet(8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    #optimizer =  RAdam(model.parameters(), lr = 0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)
    


    history = pd.DataFrame()
    
    for epoch in range(n_epochs):
        print('epoch: {}'.format(epoch+1))
        '''
        torch.cuda.empty_cache()
        gc.collect()
        '''
        train(epoch, history)
        evaluate(epoch, history)
        
    history.to_csv('history_resnet152.csv')