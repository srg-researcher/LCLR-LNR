"""
    At the first stage:
    Each local upload its current variance of loss to server
    Server monitors the average of variance and decide the time to use GMM to estimate the noisy ratio at each locals
    At the second stage:
    continue to conduct the usual label noise robust learning
"""
import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset, make_evaluate_data_loader
from fed import Federation
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger

from collections import Counter
from sklearn.mixture import GaussianMixture
import random
import json
import dataloader_cifar as dataloader

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}



def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    print('noisy_ratio', cfg['noisy_ratio'], flush=True)
    generate_noisy_ratio()
    frac_given = cfg['frac']  # for the second stage
    cfg['frac'] = 1  # for the first stage

    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']), flush=True)
        runExperiment(frac_given)
    return


def runExperiment(frac_given):
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'], cfg['subset_ratio'], cfg['noisy_ratio'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    global_parameters = model.state_dict()
    federation = Federation(global_parameters, cfg['model_rate'], label_split)
    # initial information for sampling

    sam_info = {'historical_loss': [], 'gmm_loss': [], 'entropy_loss': [],
                'max_variance': 0.0, 'continuous_quantity': 0,
                'clean_info': {'index': [], 'loss': [], 'entropy': []},
                'noise_info': {'index': [], 'loss': [], 'entropy': []},
                'curriculum': 0,
                'estimated_noisy_ratio': -1.0,
                'init_parameters': None
                }
    sam_info_list = [copy.deepcopy(sam_info) for i in range(cfg['num_users'])]

    continuous_quantity_server = 0
    max_variance_server = 0
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        # for the first stage
        # in this stage the epoch doesn't increase, but the learning rate maybe decrease
        first_stage_epoch = epoch
        while continuous_quantity_server != -1:
            sam_info_list = train(dataset['train'], data_split['train'], label_split, federation, model, optimizer,
                                  logger,
                                  first_stage_epoch, sam_info_list)
            # 1. calculate the average variance
            variance_list = []
            for i in range(len(sam_info_list)):
                variance_list.append(sam_info_list[i]['max_variance'])
            average_variance = sum(variance_list) / len(variance_list)
            if average_variance >= max_variance_server:
                max_variance_server = average_variance
                continuous_quantity_server = 0
            else:
                continuous_quantity_server += 1
            print("continuous_quantity_server: ", continuous_quantity_server)
            if continuous_quantity_server >= cfg['variance_window']:
                for i in range(len(sam_info_list)):
                    sam_info_list[i]['continuous_quantity'] = continuous_quantity_server
                sam_info_list = train(dataset['train'], data_split['train'], label_split, federation, model, optimizer,
                                      logger,
                                      first_stage_epoch, sam_info_list) # only for estimated the noisy rate
                continuous_quantity_server = -1
                # 2. recover to the normal train / second stage
                cfg['frac'] = frac_given
                # learn from the initialized parameters
                # federation = Federation(global_parameters, cfg['model_rate'], label_split)
            first_stage_epoch += 1
        # normal train / second stage
        sam_info_list = train(dataset['train'], data_split['train'], label_split, federation, model, optimizer,
                              logger,
                              epoch, sam_info_list)
        test_model = stats(dataset['train'], model)
        test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()

        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)

    print('estimated_noisy_ratio:')
    for i in range(cfg['num_users']):
        print('client {:d} estimated noisy rate {:.4f}'.format(i, sam_info_list[i]['estimated_noisy_ratio']))

    return


def generate_noisy_ratio():
    if cfg['noisy_ratio'] == -1 or cfg['noisy_ratio'] == '-1':
        cfg['noisy_ratio'] = []
        for i in range(cfg['num_users']):
            cfg['noisy_ratio'].append(-1)
    else:
        noisy_ratios = str(cfg['noisy_ratio']).split('-')
        noisy_ratios = [float(noisy_ratios[i]) for i in range(len(noisy_ratios))]
        cfg['noisy_ratio'] = []
        for i in range(cfg['num_users']):
            cfg['noisy_ratio'].append(noisy_ratios[((i + 1) % len(noisy_ratios))])
    print('generated noisy ratios: ', cfg['noisy_ratio'])


def generate_noisy_ratio_fedcorr():
    if cfg['noisy_ratio'] == -1 or cfg['noisy_ratio'] == '-1':
        cfg['noisy_ratio'] = []
        for i in range(cfg['num_users']):
            cfg['noisy_ratio'].append(-1)
    else:
        np.random.seed(1)
        gamma_s = np.random.binomial(1, cfg['level_n_system'], cfg['num_users'])
        gamma_c_initial = np.random.rand(cfg['num_users'])
        gamma_c_initial = (1 - cfg['level_n_lowerb']) * gamma_c_initial + cfg['level_n_lowerb']
        gamma_c = gamma_s * gamma_c_initial

        cfg['noisy_ratio'] = gamma_c
        print('generated noisy ratios: ', cfg['noisy_ratio'])


def train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, sam_info_list):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    start_time = time.time()
    for m in range(num_active_users):
        local_parameters[m], sam_info = copy.deepcopy(local[m].train(local_parameters[m], lr, logger, user_idx[m], epoch, dataset, sam_info_list[user_idx[m]]))
        sam_info_list[user_idx[m]] = sam_info
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                             'Learning rate: {}'.format(lr),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])
    federation.combine(local_parameters, param_idx, user_idx)
    global_model.load_state_dict(federation.global_parameters)
    
    return sam_info_list


def stats(dataset, model):
    with torch.no_grad():
        test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"])'
                          .format(cfg['model_name']))
        test_model.load_state_dict(model.state_dict(), strict=False)
        data_loader = make_data_loader({'train': dataset})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def test(dataset, data_split, label_split, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for m in range(cfg['num_users']):
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                output = model(input)   # unreasonable? should it use different local model???
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
        data_loader = make_data_loader({'test': dataset})['test']
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return


def get_initial_parameters(federation, sam_info_list):
    user_idx = torch.arange(cfg['num_users']).tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    for m in user_idx:
        sam_info_list[m]['init_parameters'] = local_parameters[m]

    return sam_info_list


def make_local(dataset, data_split, label_split, federation):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]

    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        # make noise dataset
        dataset_m = SplitDataset(dataset, data_split[user_idx[m]])
        # np.savez('data_split.npz', data_split=data_split[user_idx[m]])

        # label noise ---- start
        noisy_indices = []
        if cfg['noisy_ratio'][user_idx[m]] > 0:
            train_data_size = np.shape(dataset_m.idx)[0]
            noisy_samples_size = int(train_data_size * cfg['noisy_ratio'][user_idx[m]])
            random.seed(0)  # for repeat the experiment the same seed for generating the same indices
            noisy_indices = random.sample(range(0, train_data_size), noisy_samples_size)  # generate non-repeat numbers

            if cfg['noise_mode'] == 'sym':
                for index in noisy_indices:
                    original_label = dataset.target[dataset_m.idx[index]]
                    options = [option for option in range(10) if option != original_label]
                    random.seed(index)
                    flipped_label = random.choice(options)
                    dataset.target[dataset_m.idx[index]] = flipped_label

            if cfg['noise_mode'] == 'asym':
                transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
                for index in noisy_indices:
                    original_label = dataset.target[dataset_m.idx[index]]
                    flipped_label = transition[original_label]
                    dataset.target[dataset_m.idx[index]] = flipped_label

            if cfg['noise_mode'] == 'pairflip':
                for index in noisy_indices:
                    original_label = dataset.target[dataset_m.idx[index]]
                    flipped_label = (int(original_label) - 1) % 10
                    dataset.target[dataset_m.idx[index]] = flipped_label

        # label noise ---- end

        data_loader_m = make_data_loader({'train': dataset_m})['train']
        evaluate_data_loader_m = make_evaluate_data_loader({'train': dataset_m})['train']
        local[m] = Local(model_rate_m, data_loader_m, evaluate_data_loader_m, label_split[user_idx[m]], data_split[user_idx[m]])

    return local, local_parameters, user_idx, param_idx


def predict_noise_ratio(losses, window_length, mode='average'):
    predict_ratio = []
    if window_length > 5:
        window_length = 5
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

    return final_result


def filter_clean_data_fluctuate(epoch, sam_info):
    # 1.a calculate average of loss and entropy
    # loss_list = sam_info_list[user_index]['historical_loss'][-1].numpy()
    loss_list = sam_info['historical_loss'][-1]
    historical_loss = sam_info['historical_loss'][-cfg['window_length']:]
    # 2.a get window loss
    if epoch == 1:
        std_sigmoid = 1
    else:
        std = np.std(historical_loss, axis=0, ddof=1)
        std_sigmoid = 1 / (1 + np.exp(-std))
    window_loss = std_sigmoid * loss_list

    sample_scores = window_loss

    # 3. curriculum learning
    total_remain_k = int((1 - sam_info['estimated_noisy_ratio']) * len(loss_list))

    # 3.2 pacing function
    # the final objective is to find the number of k
    z = int((epoch) / cfg['step_length'])
    k = int(min(cfg['starting_percent'] * cfg['increase'] ** z, 1) * total_remain_k)

    idx = list(np.array(sample_scores).argsort()[:k])

    return idx, sample_scores


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return 30 * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)

criterion = SemiLoss()


class Local:
    def __init__(self, model_rate, data_loader, evaluate_data_loader, label_split, data_split_idx):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        self.evaluate_data_loader = evaluate_data_loader
        self.data_split_idx = data_split_idx

    def local_evaluate(self, model, m, epoch, sam_info):
        total = 0
        num_samples = len(self.evaluate_data_loader.dataset)
        loss_list = torch.zeros(num_samples)
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(self.evaluate_data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                output = model(input, reduction=False)
                batch_size = input['img'].size(0)
                loss_list[int(i * batch_size):int((i + 1) * batch_size)] = output['loss']
                total += batch_size
        # calculate loss variance
        loss_list = loss_list.numpy()
        loss_train = np.mean(loss_list)

        squared_diff_loss = [(x - loss_train) ** 2 for x in loss_list]
        variance_loss = sum(squared_diff_loss) / len(loss_list)

        if sam_info['estimated_noisy_ratio'] < 0.0:
            if len(sam_info['gmm_loss']) > cfg['variance_window']:  # because epoch start from 1
                sam_info['gmm_loss'].pop(0)
            sam_info['gmm_loss'].append(loss_list)
            sam_info['max_variance'] = variance_loss
        else:
            if len(sam_info['historical_loss']) > cfg['window_length']:  # because epoch start from 1
                sam_info['historical_loss'].pop(0)
            sam_info['historical_loss'].append(loss_list)

        return sam_info

    def make_sam_dataloader(self, idx, dataset):
        selected_idx = []
        for i in idx:
            selected_idx.append(self.data_split_idx[i])
        sam_dataloader = make_data_loader({'train': SplitDataset(dataset, selected_idx)})['train']

        return sam_dataloader

    def train(self, local_parameters, lr, logger, user_idx, epoch, dataset, sam_info):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))

        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)

        if sam_info['estimated_noisy_ratio'] < 0.0:
            # for the first stage
            if sam_info['continuous_quantity'] < cfg['variance_window']:
                print('Warmup Model No.%d at total epoch %d' % (user_idx, epoch))
                for local_epoch in range(1, 2):
                    for i, input in enumerate(self.data_loader):
                        input = collate(input)
                        input_size = input['img'].size(0)
                        input['label_split'] = torch.tensor(self.label_split)
                        input = to_device(input, cfg['device'])
                        optimizer.zero_grad()
                        output = model(input)
                        output['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                        logger.append(evaluation, 'train', n=input_size)

                sam_info = self.local_evaluate(model, user_idx, epoch, sam_info)
            else:
                print("No. %d Sampling started from epoch %d" % (user_idx, epoch))
                predicted_noise_ratio = predict_noise_ratio(sam_info['gmm_loss'],
                                                            cfg['variance_window'], mode='average')
                print("No. %d predicted_noise_ratio %.4f" % (user_idx, predicted_noise_ratio))
                sam_info['estimated_noisy_ratio'] = predicted_noise_ratio
                # clean hostorical loss and historical gmm loss
                sam_info['gmm_loss'] = []
                sam_info['historical_loss'] = []

        else:
            loader = dataloader.cifar_dataloader(cfg['data_name'], r=cfg['noisy_ratio'][user_idx],
                                                 noise_mode=cfg['noise_mode'],
                                                 batch_size=cfg['batch_size']['train'],
                                                 num_workers=4,
                                                 root_dir='./data/{}'.format(cfg['data_name'])
                                                 )
            for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
                # sampling
                sampling_epoch = local_epoch + (cfg['num_epochs']['local']) * (epoch - 1)  # epoch is the global epoch
                # evaluate to update loss
                self.local_evaluate(model, user_idx, sampling_epoch, sam_info)
                idx, sample_scores = filter_clean_data_fluctuate(sampling_epoch, sam_info)
                sample_ratio = float(len(idx) / len(sample_scores))
                print("client {:.4f}, estimated ratio {:.4f}, sample_ratio {:.4f} ".format(user_idx, sam_info['estimated_noisy_ratio'], sample_ratio))

                if sam_info['estimated_noisy_ratio'] <= 0.02:
                    sam_dataloader = self.make_sam_dataloader(idx, dataset)
                    # normal local train
                    for i, input in enumerate(self.data_loader):  # sam_dataloader
                        input = collate(input)
                        # MixUp
                        l = np.random.beta(4, 4)
                        l = max(l, 1 - l)

                        mixup_idx = torch.randperm(input['img'].size(0))
                        input_a, input_b = input['img'], input['img'][mixup_idx]

                        labels_x = torch.zeros(input['img'].size(0), 10).scatter_(1, input['label'].view(-1, 1), 1)
                        target_a, target_b = labels_x, labels_x[mixup_idx]

                        ## Mixup
                        mixed_input = l * input_a + (1 - l) * input_b
                        mixed_target = l * target_a + (1 - l) * target_b

                        input_mixup = {'img': None, 'label': None, 'label_split': None}
                        input_mixup['img'] = mixed_input
                        binary_label = mixed_target.topk(1, 1, True, True)[1]
                        input_mixup['label'] = binary_label.view(binary_label.size(0)).long()
                        input_size = input_mixup['img'].size(0)
                        input_mixup['label_split'] = torch.tensor(
                            self.label_split)  # label_split
                        input_mixup = to_device(input_mixup, cfg['device'])
                        optimizer.zero_grad()
                        output_mixup = model(input_mixup)
                        output_mixup['loss'].backward()
                        # output['loss'].mean().backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input_mixup,
                                                     output_mixup)
                        logger.append(evaluation, 'train', n=input_size)
                else:
                    labeled_trainloader, unlabeled_trainloader = loader.run(sample_ratio, 'train', prob=sample_scores,
                                                                            data_idx=self.data_split_idx,
                                                                            sub_dataset_rate=cfg['subset_ratio'],
                                                                            token=str(cfg['token'])+'-'+str(user_idx)
                                                                            )

                    # # train with data augmentation  --- start
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    num_iter = (len(labeled_trainloader.dataset) // cfg['batch_size']['train']) + 1

                    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x) in enumerate(
                            labeled_trainloader):
                        labels_x = labels_x.long()
                        try:
                            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
                        except:
                            unlabeled_train_iter = iter(unlabeled_trainloader)
                            inputs_u, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)

                        batch_size = inputs_x.size(0)

                        # Transform label to one-hot
                        labels_x = torch.zeros(batch_size, 10).scatter_(1, labels_x.view(-1, 1), 1)

                        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda()
                        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

                        with torch.no_grad():
                            # Label co-guessing of unlabeled samples
                            input = {'img': None, 'label': None}
                            input['img'] = inputs_u
                            input['label'] = torch.zeros(inputs_u.shape[0]).long()
                            input = to_device(input, cfg['device'])
                            outputs_u11 = model(input)
                            input['img'] = inputs_u2
                            input['label'] = torch.zeros(inputs_u2.shape[0]).long()
                            input = to_device(input, cfg['device'])
                            outputs_u12 = model(input)

                            ## Pseudo-label
                            pu = (torch.softmax(outputs_u11['score'], dim=1) + torch.softmax(outputs_u12['score'], dim=1)) / 2

                            ptu = pu ** (1 / 0.5)  ## Temparature Sharpening

                            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                            targets_u = targets_u.detach()

                            targets_x = labels_x
                            targets_x = targets_x.detach()

                        # FixMatch
                        l = np.random.beta(4, 4)
                        l = max(l, 1 - l)
                        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
                        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

                        idx = torch.randperm(all_inputs.size(0))

                        input_a, input_b = all_inputs, all_inputs[idx]
                        target_a, target_b = all_targets, all_targets[idx]

                        ## Mixup
                        mixed_input = l * input_a + (1 - l) * input_b
                        mixed_target = l * target_a + (1 - l) * target_b

                        optimizer.zero_grad()
                        input_mixup = {'img': None, 'label': None, 'label_split': None}
                        input_mixup['img'] = mixed_input
                        binary_label = mixed_target.topk(1, 1, True, True)[1]
                        input_mixup['label'] = binary_label.view(binary_label.size(0))
                        input_size = input_mixup['img'].size(0)
                        input_mixup['label_split'] = torch.tensor(self.label_split)
                        input_mixup = to_device(input_mixup, cfg['device'])
                        output_mixup = model(input_mixup)
                        logits_x = output_mixup['score'][:batch_size * 2]
                        logits_u = output_mixup['score'][batch_size * 2:]

                        ## Combined Loss
                        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u,
                                                 mixed_target[batch_size * 2:],
                                                 sampling_epoch + batch_idx / num_iter, 10)

                        loss = Lx + lamb * Lu

                        # Compute gradient and Do SGD step
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input_mixup, output_mixup)
                        logger.append(evaluation, 'train', n=input_size)
                    # # # train with data augmentation  --- end

        local_parameters = model.state_dict()
        return local_parameters, sam_info


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")