from typing import Union, Any

import numpy as np
from torch.autograd import Variable
import torch
from scipy.special import erfc


##这个文件用于生成数据，包括边界、内部、以及初始点（在这里未设置）


def fun1(z, t, ws, ds):  # 标准化变量
    stand1 = (z - ws * t) / np.sqrt(4 * ds * t)
    return stand1


def ip(z, t, ws, ds):  # 计算脉冲函数
    out = ((np.exp(-(fun1(z, t, ws, ds)) ** 2)) / (np.sqrt(np.pi * ds * t))) + erfc(fun1(z, t, ws, ds)) * ws * 0.5 / ds
    return out


def gen_label(z1, t1, ws, ds, split, p1):
    t = np.linspace(1, t1, split).reshape(1, split)
    z = np.ones_like(t) * z1
    p = np.ones_like(t).reshape(split, 1) * p1
    temp1 = (z - ws * t) / np.sqrt(4 * ds * t)
    exp1 = np.exp(-(np.square(temp1)))
    temp2 = exp1 / np.sqrt(np.pi * ds * t)
    temp3 = erfc(temp1)
    i = temp2 + (0.5 * ws / ds) * temp3
    u = np.matmul(i, p)
    return u.item()


def dt(z):
    return 1 / z


def rand_it(batch_size, variable_dim, region_a, region_b):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    x_it = x_it.astype(np.float32)
    return x_it


class DataGen2:
    def __init__(self, zs, ze, ts, te, zsteps, tsteps, ws=0.001, ds=0.0002, p=0.0001):
        super(DataGen2, self).__init__()
        self.zs = zs  # 默认值
        self.ze = ze
        self.ts = ts
        self.te = te
        self.zsteps = zsteps
        self.tsteps = tsteps
        self.ws = ws
        self.ds = ds
        self.p = p

    def gen_inter(self, batch_size):
        x_input = rand_it(batch_size, 1, self.zs, self.ze)  # 取的内部点的为随机生成
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.zeros((batch_size, 1))
        for i in range(batch_size):
            split = 100
            p = np.ones(split) * self.p
            t = np.linspace(0.1, t_input[i], split)
            y = np.zeros(split)
            pp = 0
            for j in range(split):
                temp = ip(x_input[i], t[j], self.ws, self.ds)
                y[j] = temp
            for nn in range(split):
                pp = pp + y[nn] * p[nn]
            label[i] = pp
        pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_input).float(), requires_grad=True)
        inputs_rand = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        label_rand = Variable(torch.from_numpy(label).float(), requires_grad=True)
        return inputs_rand, label_rand

    def gen_inter_m(self, batch_size):
        x_input = rand_it(batch_size, 1, self.zs, self.ze)  # 取的内部点的为随机生成
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.zeros((batch_size, 1))
        split = 100
        for i in range(batch_size):
            temp = gen_label(x_input[i], t_input[i], self.ws, self.ds, split, self.p)
            label[i] = temp
        pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_input).float(), requires_grad=True)
        inputs_rand = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        label_rand = Variable(torch.from_numpy(label).float(), requires_grad=True)
        return inputs_rand, label_rand

    def gen_bound(self, batch_size):
        x_input = rand_it(batch_size, 1, 0, 0)
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.ones(batch_size) * self.p
        pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_input).float(), requires_grad=True)
        inputs_rand = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        label_rand = Variable(torch.from_numpy(label).float(), requires_grad=True)
        return inputs_rand, label_rand

    def gen_init(self, batch_size):
        x_input = rand_it(batch_size, 1, self.zs, self.ze)
        t_input = rand_it(batch_size, 1, 0, 0)
        label = np.zeros(batch_size)
        pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_input).float(), requires_grad=True)
        inputs_rand = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        label_rand = Variable(torch.from_numpy(label).float(), requires_grad=True)
        return inputs_rand, label_rand

    def gen_labels_all(self):
        # 生成全部网格点的标签
        c = np.zeros((self.zsteps, self.tsteps))
        z = np.linspace(self.zs, self.ze, self.zsteps)
        p = np.ones(self.tsteps) * self.p
        time = np.linspace(self.ts, self.te, self.tsteps)
        for i in range(self.zsteps):  # 生成内部点
            y = np.zeros(self.tsteps)  # 这里生成一个脉冲函数的缓存数组，对于每个高度重新设置一个0向量组
            for t in range(self.tsteps):
                temp = ip(z[i], time[t], self.ws, self.ds)
                y[t] = temp
            for t in range(self.tsteps):
                pp = 0
                for nn in range(0, t + 1):
                    pp = pp + y[nn] * p[nn]
                c[i][t] = pp  # sh生成内部
        c_all = c.reshape(-1, 1)
        label1 = Variable(torch.from_numpy(c_all.copy()).float(), requires_grad=False)
        return label1

    def gen_input(self):
        x_collocation = np.linspace(self.zs, self.ze, self.zsteps).reshape(self.zsteps, 1)
        t_collocation = np.linspace(self.ts, self.te, self.tsteps).reshape(self.tsteps, 1)
        x_repeat = np.repeat(x_collocation, self.tsteps).reshape(self.zsteps * self.tsteps, 1)
        t2 = list(t_collocation)
        t1 = list(t_collocation)
        for i in range(self.zsteps - 1):
            t2.extend(t1)
        t_repeat = np.array(t2)
        pt_x_collocation = Variable(torch.from_numpy(x_repeat).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_repeat).float(), requires_grad=True)
        inputs = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        return inputs

    # 内部输入点生成
    def gen_boundary(self):
        # 边界输入点生成 z(高度)取0，时间分为steps步
        x_boundary_co = np.zeros((self.tsteps, 1))
        t_collocation = np.linspace(self.ts, self.te, self.tsteps).reshape(self.tsteps, 1)
        pt_x_boundary = Variable(torch.from_numpy(x_boundary_co).float())
        pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
        boundary_input = torch.cat([pt_t_collocation, pt_x_boundary], 1)
        return boundary_input

    def gen_boundary_label(self):
        p = np.ones(self.tsteps) * self.p

    # 当Dt是与z相关的函数
    def gen_inter_dz(self, batch_size):
        x_input = rand_it(batch_size, 1, self.zs, self.ze)
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.zeros((batch_size, 1))
        for i in range(batch_size):
            split = 100
            p = np.ones(split) * self.p
            t = np.linspace(0.1, t_input[i], split)
            y = np.zeros(split)
            temp = 0
            pp = 0
            for j in range(split):
                temp = ip(x_input[i], t[j], self.ws, self.ds_function(x_input[i]))
                y[j] = temp
            for nn in range(split):
                pp = pp + y[nn] * p[nn]
            label[i] = pp
        pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_input).float(), requires_grad=True)
        inputs_rand = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        label_rand = Variable(torch.from_numpy(label).float(), requires_grad=True)
        return inputs_rand, label_rand

    def gen_labels_all_dz(self):
        # 生成全部网格点的标签
        c = np.zeros((self.zsteps, self.tsteps))
        z = np.linspace(self.zs, self.ze, self.zsteps)
        p = np.ones(self.tsteps) * self.p
        time = np.linspace(self.ts, self.te, self.tsteps)
        for i in range(self.zsteps):  # 生成内部点
            y = np.zeros(self.tsteps)  # 这里生成一个脉冲函数的缓存数组，对于每个高度重新设置一个0向量组
            for t in range(self.tsteps):
                temp = ip(z[i], time[t], self.ws, self.ds_function(z[i]))
                y[t] = temp
            for t in range(self.tsteps):
                pp = 0
                for nn in range(0, t + 1):
                    pp = pp + y[nn] * p[nn]
                c[i][t] = pp  # sh生成内部
        c_all = c.reshape(-1, 1)
        label1 = Variable(torch.from_numpy(c_all.copy()).float(), requires_grad=False)
        return label1

    def ds_function(self, num, mode='1'):
        if mode == '1':
            return (num - self.zs) / (self.ze - self.zs) * self.ds
        elif mode == '2':
            return 1 / num
        elif mode == '3':
            return (num - self.zs) * 0.001


# u=gen_label(2, 200, 0.01, 0.01, 100, 0.001)
# print(u)
