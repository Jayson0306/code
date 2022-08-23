import numpy as np
from torch.autograd import Variable
import torch
from math import erfc


##这个文件用于生成数据，包括边界、内部、以及初始点（在这里未设置）


def fun1(z, t, ws, ds):  # 标准化变量
    stand1 = (z - ws * t) / np.sqrt(4 * ds * t)
    return stand1


def ip(z, t, ws, ds):  # 计算脉冲函数
    out = ((np.exp(-(fun1(z, t, ws, ds)) ** 2)) / (np.sqrt(np.pi * ds * t))) + erfc(fun1(z, t, ws, ds)) * ws * 0.5 / ds
    return out


class DataGen():
    def __init__(self, zs, ze, ts, te, steps, ws=0.001, ds=0.0002, p=0.0001):
        super(DataGen, self).__init__()
        self.zs = zs  # 默认值
        self.ze = ze
        self.ts = ts
        self.te = te
        self.steps = steps
        self.ws = ws
        self.ds = ds
        self.p = p

    def gen_labels_all(self):
        # 生成全部网格点的标签
        c = np.zeros((self.steps, self.steps))
        c_all = np.zeros((self.steps*self.steps, 1))
        z = np.linspace(self.zs, self.ze, self.steps)
        p = np.ones(self.steps) * self.p
        time = np.linspace(self.ts, self.te, self.steps)
        for i in range(self.steps):  # 生成内部点
            y = np.zeros(self.steps)  # 这里生成一个脉冲函数的缓存数组，对于每个高度重新设置一个0向量组
            for t in range(self.steps):
                temp = ip(z[i], time[t], self.ws, self.ds)
                y[t] = temp
            for t in range(self.steps):
                pp = 0
                for nn in range(0, t + 1):
                    pp = pp + y[nn] * p[nn]
                c[i][t] = pp  # sh生成内部
        for i in range(self.steps):
            for j in range(self.steps):
                c_all[i + j] = c[i][j]
        return c, c_all

    def gen_labels_all_diffsize(self, zsteps, tsteps):
        # 这个函数用于前期画图方便，对于高度和时间的划分可以不用同样步数，便于计算
        c = np.zeros((zsteps, tsteps))
        z = np.linspace(self.zs, self.ze, zsteps)
        p = np.ones(tsteps) * self.p
        time = np.linspace(self.ts, self.te, tsteps)
        for i in range(zsteps):  # 生成内部点
            y = np.zeros(tsteps)  # 这里生成一个脉冲函数的缓存数组，对于每个高度重新设置一个0向量组
            for t in range(tsteps):
                temp = ip(z[i], time[t], self.ws, self.ds)
                y[t] = temp
            for t in range(tsteps):  # 因为我们不需要那么多的数据，我们只取到对角线的元素即可
                pp = 0
                for nn in range(0, t + 1):
                    pp = pp + y[nn] * p[nn]
                c[i][t] = pp  # sh生成内部

        return c

    def gen_labels(self):
        # 生成内部点的函数 在神经网络中，假设高度和浓度是一一对应的，在刚刚生成的gen_labels_all 当中 我们只会用到其对角线元素
        # 在离散化积分的求解中，为了降低运行时间和成本，我们就计算到对角线元素即可
        c = np.zeros((self.steps, self.steps))
        z = np.linspace(self.zs, self.ze, self.steps)
        p = np.ones(self.steps) * self.p
        time = np.linspace(self.ts, self.te, self.steps)
        for i in range(self.steps):  # 生成内部点
            y = np.zeros(self.steps)  # 这里生成一个脉冲函数的缓存数组，对于每个高度重新设置一个0向量组
            for t in range(self.steps):
                temp = ip(z[i], time[t], self.ws, self.ds)
                y[t] = temp
            for t in range(i + 1):  # 因为我们不需要那么多的数据，我们只取到对角线的元素即可
                pp = 0
                for nn in range(0, t + 1):
                    pp = pp + y[nn] * p[nn]
                c[i][t] = pp  # sh生成内部
        data_interior = c.diagonal().reshape(self.steps, 1)  # 真实数据的interior 对角矩阵
        pde_zeros = np.zeros((self.steps, 1))
        data_boundary = self.p * np.ones(self.steps).reshape(self.steps, 1)
        pt_interior_label = Variable(torch.from_numpy(data_interior.copy()).float(), requires_grad=False)
        pt_pde_zeros = Variable(torch.from_numpy(pde_zeros).float(), requires_grad=False)
        pt_boundary_label = Variable(torch.from_numpy(data_boundary).float(), requires_grad=True)
        # labels = torch.cat([pt_pde_zeros, pt_interior_label, pt_boundary_label], 1)
        # pt_pde_zeros: pde方程的值 应当为0
        # pt_interior_label: u(x,t)的内部的真实值
        # pt_boundary_label: u(x,t)在边界的真实值
        return pt_interior_label, data_interior, pt_boundary_label

    def gen_input(self):
        x_collocation = np.linspace(self.zs, self.ze, self.steps).reshape(self.steps, 1)

        t_collocation = np.linspace(self.ts, self.te, self.steps).reshape(self.steps, 1)
        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
        interior_input = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        inputs = interior_input
        return inputs

    def gen_input2(self):
        x_collocation = np.linspace(self.zs, self.ze, self.steps).reshape(self.steps, 1)
        t_collocation = np.linspace(self.ts, self.te, self.steps).reshape(self.steps, 1)

        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
        interior_input = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        inputs = interior_input
        return inputs
    # 内部输入点生成
    def gen_boundary(self):
        # 边界输入点生成 z(高度)取0，时间分为steps步
        x_boundary_co = np.zeros((self.steps, 1))
        t_collocation = np.linspace(self.ts, self.te, self.steps).reshape(self.steps, 1)
        pt_x_boundary = Variable(torch.from_numpy(x_boundary_co).float())
        pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
        boundary_input = torch.cat([pt_t_collocation, pt_x_boundary], 1)
        return boundary_input
