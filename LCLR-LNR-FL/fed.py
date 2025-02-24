import copy
import torch
import numpy as np
from config import cfg
from collections import OrderedDict


class Federation:
    def __init__(self, global_parameters, rate, label_split):
        self.global_parameters = global_parameters
        self.rate = rate
        self.label_split = label_split
        self.make_model_rate()

    def make_model_rate(self):
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = torch.multinomial(torch.tensor(cfg['proportion']), num_samples=cfg['num_users'],
                                         replacement=True).tolist()
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)  # the ratio of subsections
        else:
            raise ValueError('Not valid model split mode')
        return

    def split_model(self, user_idx, global_epoch=0):  # 计算每一个模型参数的索引，方便后续的均值计算？
        if cfg['model_name'] == 'conv' or cfg['model_name'] == 'fcn':
            idx_i = [None for _ in range(len(user_idx))]  # 生成一个和用户数相同大小的列表吗
            idx = [OrderedDict() for _ in range(len(user_idx))]  # 生成对应客户端数的有序字典
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]  # 找到最后一个输出层的名字
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():  # 打印名字和参数  是一层一层的输的吗？
                parameter_type = k.split('.')[-1]  # 获得参数的类型
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)  #
                                output_size = v.size(0)
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)  # idx存的是索引值
                                input_idx_i_m = idx_i[m]
                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)  # 最后一个输出层的话原样输出
                                else:  # 不是最后一个输出层的话
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']  # 我就是不明白为什么要除以global rate
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]  # 后面那一层的索引值# 从这里开始对矩阵进行了裁剪，input的大小为什么没有变呢？/ 每次只改后面的这一层是对的
                                idx[m][k] = output_idx_i_m, input_idx_i_m  # 第m个local model的第k层权重矩阵
                                idx_i[m] = output_idx_i_m  # 记录下这一个矩阵的大小，用于下一个矩阵tuple的生成
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            if k == output_bias_name:  # 为什么bias不需要裁剪？
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]  # bias也切，bias就等于它对应的上一个隐层的节点数， 最后一个output层就不变嘛
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif cfg['model_name'] == 'fcn_roll':
            idx_i = [None for _ in range(len(user_idx))]  # 生成一个和用户数相同大小的列表吗
            idx = [OrderedDict() for _ in range(len(user_idx))]  # 生成对应客户端数的有序字典
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]  # 找到最后一个输出层的名字
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():  # 打印名字和参数  是一层一层的输的吗？
                parameter_type = k.split('.')[-1]  # 获得参数的类型
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)  #
                                output_size = v.size(0)
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)  # idx存的是索引值
                                input_idx_i_m = idx_i[m]
                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)  # 最后一个输出层的话原样输出
                                else:
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']  # 我就是不明白为什么要除以global rate
                                    local_output_size = int(np.ceil(output_size * scaler_rate))

                                    roll = global_epoch % output_size
                                    model_idx = torch.arange(output_size, device=v.device)
                                    model_idx = torch.roll(model_idx, roll, -1)
                                    output_idx_i_m = model_idx[:local_output_size]

                                idx[m][k] = output_idx_i_m, input_idx_i_m  # 第m个local model的第k层权重矩阵
                                idx_i[m] = output_idx_i_m  # 记录下这一个矩阵的大小，用于下一个矩阵tuple的生成
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            if k == output_bias_name:  # 为什么bias不需要裁剪？
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]  # bias也切，bias就等于它对应的上一个隐层的节点数， 最后一个output层就不变嘛
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif 'resnet' in cfg['model_name']:
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'conv1' in k or 'conv2' in k:
                                    if idx_i[m] is None:
                                        idx_i[m] = torch.arange(input_size, device=v.device)
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                elif 'shortcut' in k:
                                    input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                    output_idx_i_m = idx_i[m]
                                elif 'linear' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)  # 为什么这里linear就不切割
                                else:
                                    raise ValueError('Not valid k')
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)  # m是客户端，k是每一层，但是为什么没有乘以相应的ratio呢？
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'linear' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        elif cfg['model_name'] == 'transformer':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if 'embedding' in k.split('.')[-2]:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_input_size = int(np.ceil(input_size * scaler_rate))
                                    input_idx_i_m = torch.arange(input_size, device=v.device)[:local_input_size]
                                    idx_i[m] = input_idx_i_m
                                elif 'decoder' in k and 'linear2' in k:
                                    input_idx_i_m = idx_i[m]
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size // cfg['transformer']['num_heads']
                                                                    * scaler_rate))
                                    output_idx_i_m = (torch.arange(output_size, device=v.device).reshape(
                                        cfg['transformer']['num_heads'], -1))[:, :local_output_size].reshape(-1)
                                    idx_i[m] = output_idx_i_m
                                else:
                                    input_idx_i_m = idx_i[m]
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                    idx_i[m] = output_idx_i_m
                                idx[m][k] = (output_idx_i_m, input_idx_i_m)
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            input_size = v.size(0)
                            if 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                idx[m][k] = input_idx_i_m
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                                if 'linear_v' not in k:
                                    idx_i[m] = idx[m][k.replace('bias', 'weight')][1]
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        else:
            raise ValueError('Not valid model name')
        return idx

    def distribute(self, user_idx, global_epoch=0):  # 每一轮让local model加载当前的参数
        self.make_model_rate()  # 这个比率到底有几个，所有的local model都只服从一个比率吗
        # param_idx = self.split_model(user_idx, global_epoch)  # 返回了每一个模型的参数的索引，方便去取均值？global_epoch for FedRoleX
        param_idx = self.split_model(user_idx)  # 返回了每一个模型的参数的索引，方便去取均值？
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]  # 生成一个有序的字典
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):  # 没太看明白，这里是拷贝参数的值吗？长度不一样，是怎么拷贝上去的呢？
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])  # 把对应索引位置的值传给它
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx

    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        if cfg['model_name'] == 'conv' or cfg['model_name'] == 'fcn' or cfg['model_name'] == 'fcn_roll':
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)  # 与参数矩阵相同的全是0的值
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):  # m是local model 索引
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if k == output_weight_name:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]  # torch.meshgrid没理解是怎么回事
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]  # 这里就能实现对应位置相乘了？怎么样就生成了对应idx的坐标点？
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if k == output_bias_name:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])  # 在这个地方相除取的均值， 自己该怎么重写，全连接到底都有那些参数
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)  # 但是它也没有返回值，是把全局参数存在哪里了呢？我怀疑这是个for k, v in self.global_parameters.items(): 浅拷贝
        elif 'resnet' in cfg['model_name']:
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'linear' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]  # 就这样累加吗？那要怎么取平均呢
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1  # 是在这里计数，最后用来除吗？
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'linear' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif cfg['model_name'] == 'transformer':
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                if k.split('.')[-2] == 'embedding':
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                elif 'decoder' in k and 'linear2' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if 'decoder' in k and 'linear2' in k:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        else:
            raise ValueError('Not valid model name')

        return