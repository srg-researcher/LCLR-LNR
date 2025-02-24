import copy

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
from Asymmetric_Noise import *
from sklearn.metrics import confusion_matrix
import datasets




## If you want to use the weights and biases 
# import wandb
# wandb.init(project="noisy-label-project", entity="....")


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset): 
    def __init__(self, dataset, sample_ratio, r, noise_mode, root_dir, transform, mode, pred=[], probability=[], data_idx=[], sub_dataset_rate=-1, subset='label', token=''):
        self.r = r # noise ratio
        self.sample_ratio = sample_ratio
        self.transform = transform
        self.mode = mode

        if dataset == 'CIFAR10':
            num_class =10
        else:
            num_class =100

        ## For Asymmetric Noise (CIFAR10)    
        if noise_mode == 'asym':
            self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}
        elif noise_mode == 'asymn':
            self.transition = {0:2,2:0,4:7,7:4,1:9,9:1,3:5,5:3,6:6,8:8}

        num_sample     = -1
        self.class_ind = {}
        data = {}
        root = './data/{}'.format(dataset)
        if self.mode=='test':
            if dataset=='CIFAR10':
                data['test'] = datasets.CIFAR10(root=root, split='test', subset=subset, transform=datasets.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
                self.test_data = data['test'].img
                self.test_label = data['test'].target
            elif dataset=='cifar100':
                root_dir = './data/cifar100/'
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        
        else:    
            train_data=[]
            train_label=[]
            if dataset=='CIFAR10':
                data['train'] = datasets.CIFAR10(root=root, split='train', subset=subset, transform=datasets.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
                train_label = data['train'].target
                train_data = data['train'].img
            elif dataset=='cifar100':
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))

            # select a sub set
            if sub_dataset_rate > 0:
                train_data_size = np.shape(train_data)[0]
                subset_size = int(train_data_size * sub_dataset_rate)
                random.seed(555)  # for repeat the experiment the same seed for generating the same indices
                subset_indices = random.sample(range(0, train_data_size), subset_size)  # generate non-repeat numbers

                train_label = [train_label[i] for i in subset_indices]
                train_data = train_data[subset_indices]


            # select local dataset:
            train_label = [train_label[i] for i in data_idx]
            train_data = train_data[data_idx]
            num_sample_local = len(data_idx)

            noise_label = copy.deepcopy(train_label)
            # idx = list(range(np.shape(train_data)[0]))
            # random.shuffle(idx)
            # num_noise = int(self.r*np.shape(train_data)[0])
            # noise_idx = idx[:num_noise]

            clean_idx_in_noises = []

            # make noisy label
            if noise_mode == 'asym' or noise_mode == 'asymn':
                if dataset== 'cifar100':
                    noise_label, prob11 =  noisify_cifar100_asymmetric(train_label, self.r)
                else:
                    noisy_samples_size = int(num_sample_local * self.r)
                    random.seed(0)  # for repeat the experiment the same seed for generating the same indices
                    noisy_indices = random.sample(range(0, num_sample_local),
                                                  noisy_samples_size)  # generate non-repeat numbers
                    for index in noisy_indices:
                        original_label = train_label[index]
                        flipped_label = self.transition[original_label]
                        noise_label[index] = flipped_label
            elif noise_mode == 'pairflip':
                noisy_samples_size = int(num_sample_local * self.r)
                random.seed(0)  # for repeat the experiment the same seed for generating the same indices
                noisy_indices = random.sample(range(0, num_sample_local),
                                              noisy_samples_size)  # generate non-repeat numbers
                for index in noisy_indices:
                    original_label = train_label[index]
                    flipped_label = (int(original_label) - 1) % num_class
                    noise_label[index] = flipped_label
            else:
                if self.r > 0:
                    noisy_samples_size = int(num_sample_local * self.r)
                    random.seed(0)  # for repeat the experiment the same seed for generating the same indices
                    noisy_indices = random.sample(range(0, num_sample_local),
                                                  noisy_samples_size)  # generate non-repeat numbers
                    for index in noisy_indices:
                        original_label = train_label[index]
                        options = [option for option in range(num_class) if option != original_label]
                        random.seed(index)
                        flipped_label = random.choice(options)
                        noise_label[index] = flipped_label

            for kk in range(num_class):
                self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]  # find all indices of each class

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label

            else:
                save_file = 'Clean_index_'+str(token)+'_'+ str(dataset) + '_' +str(noise_mode) +'_' + str(self.r) + '.npz'
                save_file = os.path.join('./', save_file)

                if self.mode == "labeled":
                    pred_idx  = np.zeros(int(self.sample_ratio*num_sample_local))
                    class_len = int(self.sample_ratio*num_sample_local/num_class)
                    size_pred = 0

                    ## Ranking-based Selection and Introducing Class Balance
                    for i in range(num_class):
                        class_indices = self.class_ind[i]
                        # prob1  = np.argsort(probability[class_indices].cpu().numpy())
                        prob1  = np.argsort(probability[class_indices])
                        size1 = len(class_indices)

                        try:
                            pred_idx[size_pred:size_pred+class_len] = np.array(class_indices)[prob1[0:class_len].astype(int)].squeeze()
                            size_pred += class_len
                        except:
                            pred_idx[size_pred:size_pred+size1] = np.array(class_indices)
                            size_pred += size1

                    pred_idx = [int(x) for x in list(pred_idx)]
                    np.savez(save_file, index = pred_idx)


                    ## Weights for label refinement Ying -> for our method, we didn't use the probability
                    # probability[probability<0.5] = 0  # 为什么以0.5为界，如果不反转，又会怎么样呢？？ 这个地方不对，对UNICON都会造成影响，需要两边都要再检查一下。
                    # self.probability = [1-probability[i] for i in pred_idx] # 但是我觉得根本就不会用到这里，应该不会有影响吧？？？
                    # self.probability = [0 for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = np.load(save_file)['index']
                    idx = list(range(num_sample_local))
                    pred_idx_noisy = [x for x in idx if x not in pred_idx]
                    pred_idx = pred_idx_noisy


                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                                 

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target= self.train_data[index], self.noise_label[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4,  target

        elif self.mode=='unlabeled':
            img = self.train_data[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4

        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index

        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)   


class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir

        if self.dataset=='CIFAR10':
            transform_aw = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            self.transforms = {
                "aw_sampling": transform_aw,
                "warmup": transform_weak_10,
                "unlabeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "labeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])

        elif self.dataset=='cifar100':
            transform_aw = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            self.transforms = {
                "aw_sampling": transform_aw,
                "warmup": transform_weak_100,
                "unlabeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
                "labeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
            }        
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
                   
    def run(self, sample_ratio, mode, pred=[], prob=[], data_idx=[], sub_dataset_rate=-1,token=''):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["warmup"], mode="all", data_idx=data_idx, sub_dataset_rate=sub_dataset_rate)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,  # True
                num_workers=0, drop_last=True)  # num_workers=self.num_workers
            return trainloader

        elif mode=='no_augmentation':  # just use the aw_sampling for no augmentation setting
            all_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["aw_sampling"], mode="all", pred=pred)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=False,  # True
                num_workers=self.num_workers)
            return trainloader

        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["labeled"], mode="labeled", pred=pred, probability=prob, data_idx=data_idx, sub_dataset_rate=sub_dataset_rate,token=token)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,  # batch_size=self.batch_size
                shuffle=True,  # True
                num_workers=self.num_workers, drop_last=True)    # num_workers=self.num_workers  , drop_last=True

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["unlabeled"], mode="unlabeled", pred=pred, data_idx=data_idx, sub_dataset_rate=sub_dataset_rate,token=token)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size= int(self.batch_size/(2*sample_ratio)),  # int(self.batch_size/(2*sample_ratio))
                shuffle=True,
                num_workers=self.num_workers, drop_last =True)   # num_workers=self.num_workers

            return labeled_trainloader, unlabeled_trainloader                
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers, drop_last= True)   # 为什么要drop last 呀？
            return eval_loader        
