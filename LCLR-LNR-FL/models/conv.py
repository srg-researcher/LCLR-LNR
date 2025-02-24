import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param
from modules import Scaler


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1, track=False):
        super().__init__()
        if cfg['norm'] == 'bn':  # 是指每一层标准化的方式吗
            norm = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        elif cfg['norm'] == 'in':
            norm = nn.GroupNorm(hidden_size[0], hidden_size[0])
        elif cfg['norm'] == 'ln':
            norm = nn.GroupNorm(1, hidden_size[0])
        elif cfg['norm'] == 'gn':
            norm = nn.GroupNorm(4, hidden_size[0])
        elif cfg['norm'] == 'none':
            norm = nn.Identity()
        else:
            raise ValueError('Not valid norm')
        if cfg['scale']:
            scaler = Scaler(rate)  # 就是每一层，除以一个比率吗？
        else:
            scaler = nn.Identity()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  scaler,
                  norm,
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            if cfg['norm'] == 'bn':
                norm = nn.BatchNorm2d(hidden_size[i + 1], momentum=None, track_running_stats=track)
            elif cfg['norm'] == 'in':
                norm = nn.GroupNorm(hidden_size[i + 1], hidden_size[i + 1])
            elif cfg['norm'] == 'ln':
                norm = nn.GroupNorm(1, hidden_size[i + 1])
            elif cfg['norm'] == 'gn':
                norm = nn.GroupNorm(4, hidden_size[i + 1])
            elif cfg['norm'] == 'none':
                norm = nn.Identity()
            else:
                raise ValueError('Not valid norm')
            if cfg['scale']:
                scaler = Scaler(rate)
            else:
                scaler = nn.Identity()
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           scaler,
                           norm,
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(hidden_size[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)  # 创建了CNN模型，最后一层是扁平化， 我现在不太清楚，这是一个模型还是一组模型

    def forward(self, input, reduction=True):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        out = self.blocks(x)
        if 'label_split' in input and cfg['mask']:
            label_mask = torch.zeros(cfg['classes_size'], device=out.device)
            label_mask[input['label_split']] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output['score'] = out
        if reduction == False:
            output['loss'] = F.cross_entropy(out, input['label'], reduction='none')
        else:
            output['loss'] = F.cross_entropy(out, input['label'], reduction='mean')
        return output


def conv(model_rate=1, track=False):
    data_shape = cfg['data_shape']  # [1, 28, 28]
    hidden_size = [int(np.ceil(model_rate * x)) for x in cfg['conv']['hidden_size']]  # 就一个model rate吗？不应该是不同的模型形态吗？'hidden_size': [64, 128, 256, 512]
    classes_size = cfg['classes_size']  # 在MNIST的时候赋予的
    scaler_rate = model_rate / cfg['global_model_rate']  # global model rate 到底是什么意思，为什么要去除这个？
    model = Conv(data_shape, hidden_size, classes_size, scaler_rate, track)  # 会生成一组模型还是一个模型
    model.apply(init_param)  # 初始化模型参数，为什么全填1或0？
    return model