from __future__ import print_function
import sys

import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
# from resnet18 import *
import dataloader_cifar as dataloader
from math import log2

import collections.abc
from collections.abc import MutableMapping

from collections import Counter
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.mixture import GaussianMixture
import copy
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import sys
import time

# Ying: delete redundant codes
# for symmetric first window size equals to 5

## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project", entity="..")

## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u', default=0.7, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default='JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help='Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--window_length', default=3, type=int, help='window length for loss of aw')
parser.add_argument('--variance_window', default=5, type=int, help='window length for warm start of aw')
parser.add_argument('--sampling_epoch', default=0, type=int, help='the epoch of starting sampling of aw')
parser.add_argument('--starting_percent', default=0.35, type=float, help='starting percent for curriculum  of aw')
parser.add_argument('--step_length', default=40, type=float, help='step length for curriculum  of aw')
parser.add_argument('--increase', default=1.5, type=float, help='the increase ratio for curriculum  of aw')
parser.add_argument('--upln_epoch', default=-1, type=int, help='from when to start the semi-supervised learning  of aw')
parser.add_argument('--sampling_mode', default='aw_sampling', type=str, help='whether augmentation of aw')
parser.add_argument('--contrastive_loss', default=0, type=int, help='whether use cl of aw, 1 use; 0 no use')

args = parser.parse_args()

## GPU Setup
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


## Download the Datasets
if args.dataset == 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path, train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path, train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path, train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path, train=False, download=True)

## Checkpoint Location
folder = 'aw_' + args.dataset + '_' + args.noise_mode + '_' + str(args.r)
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## Log files
stats_log = open(model_save_loc + '/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')
test_log = open(model_save_loc + '/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')
test_loss_log = open(model_save_loc + '/test_loss.txt', 'w')
train_acc = open(model_save_loc + '/train_acc.txt', 'w')
train_loss = open(model_save_loc + '/train_loss.txt', 'w')


# SSL-Training
def train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_ucl = 0

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            _, outputs_u11 = net(inputs_u)
            _, outputs_u12 = net(inputs_u2)

            ## Pseudo-label
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2

            ptu = pu ** (1 / args.T)  ## Temparature Sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            targets_x = labels_x
            targets_x = targets_x.detach()


        # FixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        ## Mixup
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        _, logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        ## Combined Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)

        loss = Lx + lamb * Lu

        ## Accumulate Loss
        loss_x += Lx.item()
        loss_u += Lu.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f'
            % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
               loss_x / (batch_idx + 1), loss_u / (batch_idx + 1), loss_ucl / (batch_idx + 1)))
        sys.stdout.flush()


## For Standard Training 
def warmup_standard(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1

    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        _, outputs = net(inputs)
        loss = CEloss(outputs, labels)

        if args.noise_mode == 'asym' or args.noise_mode == 'asymn':  # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        else:
            L = loss

        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


## For Training Accuracy
def warmup_val(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            _, outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = CEloss(outputs, labels)
            loss_x += loss.item()

            total += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))

    train_loss.write(str(loss_x / (batch_idx + 1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc


## Test Accuracy for one model wrote by ying
def test_one_model(epoch, net1):
    net1.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = CEloss(outputs, targets)
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write(str(acc) + '\n')
    test_log.flush()
    test_loss_log.write(str(loss_x / (batch_idx + 1)) + '\n')
    test_loss_log.flush()
    return acc


## Unsupervised Loss coefficient adjustment
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)  # 感觉一个用的是交叉熵，一个用的是均方误差
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model


def output_mean_loss():
    noise_file = '%s/clean_%.4f_%s.npz' % (args.data_path, args.r, args.noise_mode)
    noise_idx = np.load(noise_file)['index']
    current_loss = sam_info['historical_loss'][-1]
    clean_loss = []
    noisy_loss = []
    for i in range(len(current_loss)):
        if i in noise_idx:
            noisy_loss.append(current_loss[i])
        else:
            clean_loss.append(current_loss[i])
    noisy_loss_average = np.mean(noisy_loss)
    clean_loss_average = np.mean(clean_loss)
    print('\naverage loss -> clean: {:.4f} noise: {:.4f}'.format(clean_loss_average, noisy_loss_average))


def clean_noisy_evaluation(epoch, net1, num_samples):
    net1.eval()
    correct = 0
    total = 0
    loss_list = torch.zeros(num_samples)

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            _, outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = CE(outputs, targets)

        loss_list[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)] = loss

        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

    loss_list = loss_list.numpy()
    loss_train = np.mean(loss_list)
    squared_diff_loss = [(x - loss_train) ** 2 for x in loss_list]
    variance_loss = sum(squared_diff_loss) / len(loss_list)

    if len(sam_info['historical_loss']) > args.window_length:  # because epoch start from 1
        sam_info['historical_loss'].pop(0)
    sam_info['historical_loss'].append(loss_list)
    if variance_loss >= sam_info['max_variance']:
        sam_info['max_variance'] = variance_loss
        sam_info['continuous_quantity'] = 0
        sam_info['gmm_loss'] = []
        sam_info['gmm_loss'].append(loss_list.tolist())
    else:
        sam_info['continuous_quantity'] += 1
        sam_info['gmm_loss'].append(loss_list.tolist())


def predict_noise_ratio(losses, window_length, mode='average'):
    predict_ratio = []
    first_window = 5 if args.noise_mode == 'sym' else 20
    if window_length > first_window:
        window_length = first_window
    window_length = min(window_length, first_window)
    for i in range(window_length):
        temp = copy.deepcopy(losses[i])
        random_state = 0
        gmm = GaussianMixture(n_components=2, covariance_type="tied", max_iter=5000, n_init=5,
                              random_state=random_state)
        samples = np.array(temp).reshape(-1, 1)
        gmm.fit(samples)
        y_pred_prob = gmm.predict_proba(samples)
        pred_sum = np.sum(y_pred_prob, axis=0)
        if gmm.means_[0] <= gmm.means_[1]:
            noise_ratio = pred_sum[1] / len(samples)
        else:
            noise_ratio = pred_sum[0] / len(samples)
        predict_ratio.append(noise_ratio)

    if mode == 'average':
        final_result = sum(predict_ratio) / len(predict_ratio)
    elif mode == 'first':
        final_result = predict_ratio[0]
    print('predicted noise ratio: ', final_result)

    return final_result


def filter_clean_data_fluctuate(net1, epoch, sam_info, predicted_noise_ratio):
    # 1. get loss
    loss_list = torch.zeros(num_samples)
    predicted_list = torch.zeros(num_samples)

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        with torch.no_grad():
            _, outputs = net1(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = CE(outputs, targets)

        loss_list[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)] = loss
        predicted_list[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)] = predicted

    loss_list = loss_list.numpy()
    if len(sam_info['historical_loss']) > args.window_length:  # because epoch start from 1
        sam_info['historical_loss'].pop(0)
    sam_info['historical_loss'].append(loss_list)
    if epoch == 1:
        std_sigmoid = 1
    else:
        std = np.std(sam_info['historical_loss'], axis=0, ddof=1)
        std_sigmoid = 1 / (1 + np.exp(-std))
    window_loss = std_sigmoid * loss_list
    sample_scores = window_loss

    # 3. curriculum learning
    total_remain_k = int((1 - predicted_noise_ratio) * len(loss_list))
    z = int((epoch - args.sampling_epoch) / args.step_length) 
    k = int(min(args.starting_percent * args.increase ** z, 1) * total_remain_k)
    idx = list(np.array(sample_scores).argsort()[:k])

    return idx, sam_info, sample_scores


sam_info = {'historical_loss': [], 'gmm_loss': [], 'entropy_loss': [],
            'max_variance': 0.0, 'continuous_quantity': 0,
            'clean_info': {'index': [], 'loss': [], 'entropy': []},
            'noise_info': {'index': [], 'loss': [], 'entropy': []},
            'curriculum': 0,
            'estimated_noisy_rate': -1
            }


## Choose Warmup period based on Dataset
num_samples = 50000
if args.dataset == 'cifar10':
    warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                     num_workers=4, \
                                     root_dir=model_save_loc, log=stats_log,
                                     noise_file='%s/clean_%.4f_%s.npz' % (
                                     args.data_path, args.r, args.noise_mode))

print('| Building net')
net1 = create_model()

cudnn.benchmark = True

## Semi-Supervised Loss
criterion = SemiLoss()

## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 280, 2e-4)

## Loss Functions
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss(reduction='none')


if args.noise_mode == 'asym' or args.noise_mode == 'asymn':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint 
model_name_1 = 'Net1_warmup.pth'

if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
else:
    start_epoch = 0

best_acc = 0

start_time = time.time()
test_loader = loader.run(0, 'test')
eval_loader = loader.run(0, 'eval_train')
warmup_trainloader = loader.run(0, 'warmup')

for epoch in range(start_epoch, args.num_epochs + 1):
    if sam_info['continuous_quantity'] < args.variance_window:
        print('Warmup Model')
        warmup_standard(epoch, net1, optimizer1, warmup_trainloader)
        clean_noisy_evaluation(epoch, net1, num_samples)
    else:
        if sam_info['estimated_noisy_rate'] < 0:
            args.sampling_epoch = epoch
            warm_up = epoch
            print("Sampling started from epoch ", args.sampling_epoch)
            sam_info['estimated_noisy_rate'] = predict_noise_ratio(sam_info['gmm_loss'], args.variance_window, mode='average')
            print("predicted_noise_ratio ", sam_info['estimated_noisy_rate'])

        pred_idx, sam_info, sample_scores = filter_clean_data_fluctuate(net1, epoch, sam_info, sam_info['estimated_noisy_rate'])
        pred_idx = [int(x) for x in list(pred_idx)]
        selected_idx = pred_idx
        sample_ratio = float(len(selected_idx) / num_samples)
        scores = torch.tensor(sample_scores)
        labeled_trainloader, unlabeled_trainloader = loader.run(sample_ratio, 'train', prob=scores)
        train(epoch, net1, optimizer1, labeled_trainloader, unlabeled_trainloader)

    acc = test_one_model(epoch, net1)
    scheduler1.step()

    if acc > best_acc:
        if epoch < warm_up:
            model_name_1 = 'Net1_warmup.pth'
        else:
            model_name_1 = 'Net1.pth'

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'CIFAR10',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        best_acc = acc

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")