import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import PIL.Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import albumentations as A
import cv2
import os
import csv
import pdb
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import pickle
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

class contrast_processing_rgb(ImageOnlyTransform):
    
    def __init__(self,img_size = 800, always_apply=False,p=1.0):
        super(contrast_processing_rgb, self).__init__(always_apply, p)
        b = np.zeros((img_size,img_size,3))
        self.size = img_size
        self.p = p
        self.b = cv2.circle(b, (int(self.size / 2), int(self.size / 2)),int(self.size/2), (1, 1, 1), -1, 8, 0)
    
    def apply(self, img, **params):
        img = np.asarray(img)
        #if (np.random.rand(1) < self.p):
        blur_img = cv2.GaussianBlur(img, (0, 0), self.size / 30)
        merge_img = cv2.addWeighted(img, 4, blur_img, -4, 128)
        merge_img = merge_img * self.b + 128 * (1 - self.b)
        merge_img = merge_img.astype(np.uint8)
        #else:
        #    merge_img = img
        return merge_img

def train_transform(img_size):
    return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Flip(p=0.8),
                    A.Rotate(p=0.8),
                    A.OneOf([
                       A.ShiftScaleRotate(rotate_limit=0,p=0.1),
                       A.RandomSizedCrop(min_max_height=(int(0.5*img_size), img_size), height=img_size, width=img_size, p=0.9)
                    ], p=1.0),
                    A.OneOf([
                        A.MedianBlur(blur_limit=3, p=1.0),
                        A.Sharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
                        A.CLAHE(p=1.0),
                        A.Equalize(p=1.0),
                        contrast_processing_rgb()
                    ], p=0.9),
                    A.OneOf([
                       A.GaussNoise(p=1.0),
                       A.ISONoise(p=1.0)
                    ], p=0.6),
                    A.ColorJitter(0.4,0.4,0.4,0.4,p=0.9),
                    A.Normalize(),
                    A.CoarseDropout(),
                    ToTensorV2(),
                ]
            )

def eval_transform(img_size):
     return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
    
class medicalDataset(torch.utils.data.Dataset):
    def __init__(self,epoch,split,train=True):
        super(medicalDataset, self).__init__()
        if train:
            self.data = (mmac_task1_train_dataset(split)*2+mmac_task2_train_dataset(split)*2#+mmac_task3_train_dataset(split)
            +airogs_task1_pseudo_dataset(split,0)+palm_training_task1_pseudo_dataset(split,0)+palm_validation_task1_pseudo_dataset(split,0)+mmac_task2_cutmix_dataset(split,0))
            #+airogs_task1_pseudo_dataset(0)+dr_kaggle_test_task1_pseudo_dataset(0)+dr_kaggle_train_task1_pseudo_dataset(0))
            self.transform = train_transform(800)
        else:
            self.data = (mmac_task1_test_dataset(split)+mmac_task2_test_dataset(split)+mmac_task3_test_dataset(split))
            self.transform = eval_transform(800)
        self.train = train
        self.split = split
    
    def resample(self,epoch):
        print('resample')
        self.data = (mmac_task1_train_dataset(self.split)*2+mmac_task2_train_dataset(self.split)*2#+mmac_task3_train_dataset(self.split)
            +airogs_task1_pseudo_dataset(self.split,epoch)+palm_training_task1_pseudo_dataset(self.split,epoch)+palm_validation_task1_pseudo_dataset(self.split,epoch)+mmac_task2_cutmix_dataset(self.split,epoch))
        
    def __getitem__(self,index):
        
        img_path = self.data[index]['path']
        task = self.data[index]['task']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        task2_target = None
        if task==21 or task==22 or task==23:
            task2_target = (cv2.imread(self.data[index]['target_path'],0)/255.0)
        else:
            h,w,_ = img.shape
            task2_target = np.zeros((h,w))

        transformed = self.transform(image = img, mask=task2_target)
        img = transformed['image']
        task2_target = transformed['mask']

        task1_target = torch.zeros(5)
        if task==1:
            task1_target[int(self.data[index]['target'])] = 1
        
        task3_target = torch.ones(1)
        if task==3:
            task3_target = task3_target*float(self.data[index]['target'])
        
        #task3_target_normalized = (task3_target +10)/20
        #task3_target_normalized = torch.clamp(task3_target_normalized,0,1)

        img_dict = {}
        img_dict['img'] = img
        img_dict['task1_target'] = task1_target
        img_dict['task2_target'] = task2_target.to(torch.int64)
        img_dict['task3_target'] = task3_target
        img_dict['task3_se'] = task3_target
        img_dict['path'] = img_path
        img_dict['task'] = task
    
        return img_dict

    def __len__(self):
        return len(self.data)

class medicalDataset_task1(torch.utils.data.Dataset):
    def __init__(self,epoch,split,train=True):
        super(medicalDataset_task1, self).__init__()
        if train:
            self.data = (mmac_task1_train_dataset(split)*2+airogs_task1_pseudo_dataset(split,0)+palm_training_task1_pseudo_dataset(split,0)+palm_validation_task1_pseudo_dataset(split,0))*2
            #+airogs_task1_pseudo_dataset(0)+dr_kaggle_test_task1_pseudo_dataset(0)+dr_kaggle_train_task1_pseudo_dataset(0))
            self.transform = train_transform(800)
        self.train = train
        self.split = split
    
    def resample(self,epoch):
        print('resample')
        self.data = (mmac_task1_train_dataset(self.split)*2
            +airogs_task1_pseudo_dataset(self.split,epoch)+palm_training_task1_pseudo_dataset(self.split,epoch)+palm_validation_task1_pseudo_dataset(self.split,epoch))
       
    def __getitem__(self,index):
        
        img_path = self.data[index]['path']
        task = self.data[index]['task']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image = img)
        img = transformed['image']

        task1_target = torch.zeros(5)
        task1_target[int(self.data[index]['target'])] = 1

        img_dict = {}
        img_dict['img'] = img
        img_dict['task1_target'] = task1_target
        img_dict['path'] = img_path
        img_dict['task'] = task
    
        return img_dict

    def __len__(self):
        return len(self.data)

##################################################################################################################################
def mmac_task1_test_dataset(split):
    dataset = []
    data = pd.read_csv('./csv/mmac_task1/test_split_1_'+str(split)+'.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

    data = pd.read_csv('./csv/mmac_task1/test_split_2_'+str(split)+'.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    return dataset
##################################################################################################################################
def mmac_task3_train_dataset(split):
    dataset = []                
    data = pd.read_csv('./csv/mmac_task3/train_split_'+str(split)+'.csv')
   
    num_bins = 10
    # Convert the target variable into bins
    data['target_bin'] = pd.cut(data['target'], bins=num_bins, labels=False)

    # Separate majority and minority classes
    data_dict = {}
    max_value = 0
    for i in range(num_bins):
        temp = data[data['target_bin']==i]
        data_dict[i] = len(temp)
        max_value = max(len(temp),max_value)
    max_value = max_value//2

    for k,v in data_dict.items():
        if max_value==v or v==0:
            continue
        mul = max_value//v
        temp = [data,]
        for m in range(mul):
            temp.append(data[data['target_bin']==k])
        data = pd.concat(temp)

    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task3/task3_data/1. Images/1. Training Set/'+row['image'],'target':row['target'],'age':row['age'],'sex':row['sex'],'task':3})
    
    data = pd.read_csv('./csv/mmac_task3/PAPILA_od.csv')
    data = data.dropna()

    num_bins = 10
    # Convert the target variable into bins
    data['target_bin'] = pd.cut(data['od_se'], bins=num_bins, labels=False)

    # Separate majority and minority classes
    data_dict = {}
    max_value = 0
    for i in range(num_bins):
        temp = data[data['target_bin']==i]
        data_dict[i] = len(temp)
        max_value = max(len(temp),max_value)
    max_value = max_value//2

    for k,v in data_dict.items():
        if max_value==v or v==0:
            continue
        mul = max_value//v
        temp = [data,]
        for m in range(mul):
            temp.append(data[data['target_bin']==k])
        data = pd.concat(temp)

    for _,row in data.iterrows():
        dataset.append({'path' : '/data/PAPILA/resize_fundus_images/RET'+'{0:03d}'.format((int(row['od_path'])))+'OD.jpg','target':float(row['od_se']),'age':row['age'],'sex':-1,'task':3})

    data = pd.read_csv('./csv/mmac_task3/PAPILA_os.csv')
    data = data.dropna()

    num_bins = 10
    # Convert the target variable into bins
    data['target_bin'] = pd.cut(data['os_se'], bins=num_bins, labels=False)

    # Separate majority and minority classes
    data_dict = {}
    max_value = 0
    for i in range(num_bins):
        temp = data[data['target_bin']==i]
        data_dict[i] = len(temp)
        max_value = max(len(temp),max_value)
    max_value = max_value//2

    for k,v in data_dict.items():
        if max_value==v or v==0:
            continue
        mul = max_value//v
        temp = [data,]
        for m in range(mul):
            temp.append(data[data['target_bin']==k])
        data = pd.concat(temp)

    for _,row in data.iterrows():
        dataset.append({'path' : '/data/PAPILA/resize_fundus_images/RET'+'{0:03d}'.format((int(row['os_path'])))+'OS.jpg','target':float(row['os_se']),'age':row['age'],'sex':-1,'task':3})
    return dataset
def mmac_task3_test_dataset(split):
    dataset = []
    data = pd.read_csv('./csv/mmac_task3/test_split_'+str(split)+'.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task3/task3_data/1. Images/1. Training Set/'+row['image'],'target':row['target'],'age':row['age'],'sex':row['sex'],'task':3})
    return dataset
##################################################################################################################################


def mmac_task2_cutmix_dataset(split,epoch):

    dataset = []
    for split in range(5):
        data = glob('/home/luke/cutmix_data/split'+str(split)+'/1/img/*.png')
        random.seed(epoch)
        random.shuffle(data)
        data = data[:300]
        for d in data:
            dataset.append({'path' :d,
            'target_path' : d.replace('img','mask'),'age':-1,'sex':-1,'task':21})
        
        data = glob('/home/luke/cutmix_data/split'+str(split)+'/2/img/*.png')
        random.seed(epoch)
        random.shuffle(data)
        data = data[:300]
        for d in data:
            dataset.append({'path' :d,
            'target_path' : d.replace('img','mask'),'age':-1,'sex':-1,'task':22})
        
        data = glob('/home/luke/cutmix_data/split'+str(split)+'/3/img/*.png')
        random.seed(epoch)
        random.shuffle(data)
        data = data[:300]
        for d in data:
            dataset.append({'path' :d,
            'target_path' : d.replace('img','mask'),'age':-1,'sex':-1,'task':23})
    
    return dataset

def mmac_task2_test_dataset(split):
    dataset = []                
    data = pd.read_csv('./csv/mmac_task2/test_split_'+str(split)+'_1.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/1. Lacquer Cracks/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/1. Lacquer Cracks/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':21})
               
    data = pd.read_csv('./csv/mmac_task2/test_split_'+str(split)+'_2.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':22})
               
    data = pd.read_csv('./csv/mmac_task2/test_split_'+str(split)+'_3.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/3. Fuchs Spot/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/3. Fuchs Spot/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':23})
    
    return dataset

##################################################################################################################################

def airogs_task1_pseudo_dataset(split,epoch):
    dataset = []
    
    for i in range(5):
        data = pd.read_csv('/home/luke/mmac_multitask/csv/airogs_pseudo_label/train_split_'+str(split)+'_iter1.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
        data = pd.read_csv('/home/luke/mmac_multitask/csv/airogs_pseudo_label/train_split_'+str(split)+'_iter2.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
    return dataset

def palm_training_task1_pseudo_dataset(split,epoch):
    dataset = []
    
    for i in range(5):

        data = pd.read_csv('/home/luke/mmac_multitask/csv/palm_training400/train_split_'+str(split)+'_iter1.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
    for i in range(5):

        data = pd.read_csv('/home/luke/mmac_multitask/csv/palm_training400/train_split_'+str(split)+'_iter2.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
    return dataset

def palm_validation_task1_pseudo_dataset(split,epoch):
    dataset = []
    
    for i in range(5):
        data = pd.read_csv('/home/luke/mmac_multitask/csv/palm_validation400/train_split_'+str(split)+'_iter1.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
    for i in range(5):
        data = pd.read_csv('/home/luke/mmac_multitask/csv/palm_validation400/train_split_'+str(split)+'_iter2.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
    return dataset

def dr_task1_pseudo_dataset(split,epoch):
    dataset = []
    
    for i in range(5):

        data = pd.read_csv('/home/luke/mmac_multitask/csv/dr_kaggle_test_pseudo_label.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
    for i in range(5):
        data = pd.read_csv('/home/luke/mmac_multitask/csv/dr_kaggle_train_pseudo_label.csv')
        data = data[data['target']==i]
        data = data.sample(frac=1, replace=False,random_state=epoch)
        num = 150#int(max(100,len(data)*0.05))
        data = data[:num]
        for _,row in data.iterrows():
            dataset.append({'path' : row['path'].replace('/home/sonic/','/data/'),'data_center':3,
                'target':row['target'],
            'age':-1,'sex':-1,'task':1,'target':i})
    return dataset

def palm_task2_pseudo_dataset():
    dataset = []                
    data = glob('/home/luke/mmac_task2/pseudo_mask_class1_ensamble/PALM-Validation400/*.jpg')
    for d in data:
    
        dataset.append({'path' : '/data/PALM-Validation400/'+d.split('/')[-1],
        'target_path' : '/home/luke/mmac_task2/pseudo_mask_class1_ensamble/PALM-Validation400/'+d.split('/')[-1],'age':-1,'sex':-1,'task':21})
    
    data = glob('/home/luke/mmac_task2/pseudo_mask_class2_ensamble/PALM-Validation400/*.jpg')
    for d in data:
    
        dataset.append({'path' : '/data/PALM-Validation400/'+d.split('/')[-1],
        'target_path' : '/home/luke/mmac_task2/pseudo_mask_class2_ensamble/PALM-Validation400/'+d.split('/')[-1],'age':-1,'sex':-1,'task':22})
    
    data = glob('/home/luke/mmac_task2/pseudo_mask_class3_ensamble/PALM-Validation400/*.jpg')
    for d in data:
    
        dataset.append({'path' : '/data/PALM-Validation400/'+d.split('/')[-1],
        'target_path' : '/home/luke/mmac_task2/pseudo_mask_class3_ensamble/PALM-Validation400/'+d.split('/')[-1],'age':-1,'sex':-1,'task':23})
    
    data = glob('/home/luke/mmac_task2/pseudo_mask_class1_ensamble/PALM-Training400/*.jpg')
    for d in data:
    
        dataset.append({'path' : '/data/PALM-Training400/'+d.split('/')[-1],
        'target_path' : '/home/luke/mmac_task2/pseudo_mask_class1_ensamble/PALM-Training400/'+d.split('/')[-1],'age':-1,'sex':-1,'task':21})
    
    data = glob('/home/luke/mmac_task2/pseudo_mask_class2_ensamble/PALM-Training400/*.jpg')
    for d in data:
    
        dataset.append({'path' : '/data/PALM-Training400/'+d.split('/')[-1],
        'target_path' : '/home/luke/mmac_task2/pseudo_mask_class2_ensamble/PALM-Training400/'+d.split('/')[-1],'age':-1,'sex':-1,'task':22})
    
    data = glob('/home/luke/mmac_task2/pseudo_mask_class3_ensamble/PALM-Training400/*.jpg')
    for d in data:
    
        dataset.append({'path' : '/data/PALM-Training400/'+d.split('/')[-1],
        'target_path' : '/home/luke/mmac_task2/pseudo_mask_class3_ensamble/PALM-Training400/'+d.split('/')[-1],'age':-1,'sex':-1,'task':23})
    
    print('task2 pseudo',len(dataset))
    return dataset

def mmac_task1_train_dataset(split):
    dataset = []                
    data = pd.read_csv('./csv/mmac_task1/train_split_1_'+str(split)+'.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

        #dataset.append({'path' : '/data/enhancement_data_isecret/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

        #dataset.append({'path' : '/data/enhancement_data_pcenet/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

        #dataset.append({'path' : '/data/enhancement_data_scrnet/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/train_split_1_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/train_split_1_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1 or row['target']==2:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

    data = pd.read_csv('./csv/mmac_task1/train_split_2_'+str(split)+'.csv')
    for _,row in data.iterrows():

        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

        #dataset.append({'path' : '/data/enhancement_data_isecret/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

        #dataset.append({'path' : '/data/enhancement_data_pcenet/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

        #dataset.append({'path' : '/data/enhancement_data_scrnet/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/train_split_2_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/train_split_2_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1 or row['target']==2:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

    ########

    data = pd.read_csv('./csv/mmac_task1/test_split_1_'+str(split)+'.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/test_split_1_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/test_split_1_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1 or row['target']==2:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':1, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})

    data = pd.read_csv('./csv/mmac_task1/test_split_2_'+str(split)+'.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/test_split_2_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    
    data = pd.read_csv('./csv/mmac_task1/test_split_2_'+str(split)+'.csv')
    for _,row in data.iterrows():
        if row['target']==0 or row['target']==1 or row['target']==2:
            continue
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
        dataset.append({'path' : '/home/luke/mmac_task1/task1_data_update/1. Images/1. Training Set/'+row['image'],'data_center':2, 'target':row['target'],'age':row['age'],'sex':row['sex'],'task':1})
    return dataset

def mmac_task2_train_dataset(split):
    dataset = []                
    data = pd.read_csv('./csv/mmac_task2/train_split_'+str(split)+'_1.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/1. Lacquer Cracks/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/1. Lacquer Cracks/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':21})
               
    data = pd.read_csv('./csv/mmac_task2/train_split_'+str(split)+'_2.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':22})
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':22})
               
    data = pd.read_csv('./csv/mmac_task2/train_split_'+str(split)+'_3.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/3. Fuchs Spot/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/3. Fuchs Spot/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':23})
    
    data = pd.read_csv('./csv/mmac_task2/test_split_'+str(split)+'_1.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/1. Lacquer Cracks/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/1. Lacquer Cracks/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':21})
               
    data = pd.read_csv('./csv/mmac_task2/test_split_'+str(split)+'_2.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':22})
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/2. Choroidal Neovascularization/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':22})
               
    data = pd.read_csv('./csv/mmac_task2/test_split_'+str(split)+'_3.csv')
    for _,row in data.iterrows():
        dataset.append({'path' : '/home/luke/mmac_task2/task2_data/3. Fuchs Spot/1. Images/1. Training Set/'+row['image'],
        'target_path' : '/home/luke/mmac_task2/task2_data/3. Fuchs Spot/2. Groundtruths/1. Training Set/'+row['image'],'age':row['age'],'sex':row['sex'],'task':23})
    
    return dataset

if __name__ == '__main__':
    split = 0
    data = (mmac_task1_train_dataset(split)*2+mmac_task2_train_dataset(split)*2#+mmac_task3_train_dataset(split)
            +airogs_task1_pseudo_dataset(split,0)+palm_training_task1_pseudo_dataset(split,0)+palm_validation_task1_pseudo_dataset(split,0)+mmac_task2_cutmix_dataset(split,0)+dr_task1_pseudo_dataset(split,0))
    
    data = pd.DataFrame.from_dict(data)
    print(len(data[data['target']==0]))
    print(len(data[data['target']==1]))
    print(len(data[data['target']==2]))
    print(len(data[data['target']==3]))
    print(len(data[data['target']==4]))
    

