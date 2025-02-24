import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param
from modules import Scaler


class Fcn(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size, rate=1, track=False):
        super().__init__()
        # self.flatten = nn.Flatten()

        if cfg['norm'] == 'bn':
            norm = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)  # 这个没法用
        elif cfg['norm'] == 'in':
            norm = nn.GroupNorm(hidden_size[0], hidden_size[0])  # 这个加了之后，效果反而不好
        elif cfg['norm'] == 'ln':
            norm = nn.GroupNorm(1, hidden_size[0])
        elif cfg['norm'] == 'gn':
            norm = nn.GroupNorm(4, hidden_size[0])
        elif cfg['norm'] == 'none':  # 应为模型太简单了，所以我觉得就不需要norm
            norm = nn.Identity()
        else:
            raise ValueError('Not valid norm')
        if cfg['scale']:
            scaler = Scaler(rate)
        else:
            scaler = nn.Identity()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, hidden_size[0]),
            scaler,
            norm,
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[0]),
            scaler,
            norm,
            nn.ReLU(),
            nn.Linear(hidden_size[0], classes_size),
        )

        # self.fc1 = nn.Linear(28*28, 10)
        # self.fc2 = nn.Linear(10, 10)
        # self.fc3 = nn.Linear(10, 10)

        # self.linear1 = torch.nn.Linear(19, 10)
        # self.activation = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(10, classes_size)
        # self.softmax = torch.nn.Softmax()

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input['img']
        #out = self.linear_relu_stack(x)
        x = x.view(x.size(0), -1)
        out = self.linear_relu_stack(x)

        # out = self.linear1(x)
        # out = self.activation(out)
        # out = self.linear2(out)
        # out = self.softmax(out)

        if 'label_split' in input and cfg['mask']:
            label_mask = torch.zeros(cfg['classes_size'], device=out.device)
            label_mask[input['label_split']] = 1
            out = out.masked_fill(label_mask == 0, 0)
        output['score'] = out
        output['loss'] = F.cross_entropy(out, input['label'], reduction='mean')  # 必须给logit, 不能给softmax
        # output['loss'] = F.cross_entropy(out, input['label'], reduction='sum')
        # output['loss'] = F.cross_entropy(out, input['label'])
        return output


def fcn(model_rate=1, track=False):
    data_shape = cfg['data_shape']  # [1, 28, 28]
    hidden_size = [int(np.ceil(model_rate * x)) for x in [10]]  # 就一个model rate吗？不应该是不同的模型形态吗？'hidden_size': [64, 128, 256, 512]
    classes_size = cfg['classes_size']  # 在MNIST的时候赋予的
    scaler_rate = model_rate / cfg['global_model_rate']  # 用于scale
    model = Fcn(data_shape, hidden_size, classes_size, scaler_rate, track)  # 会生成一组模型还是一个模型
    model.apply(init_param)  # 初始化模型参数，为什么全填1或0？
    return model