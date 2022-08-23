import torch
import torch.nn as tn
from pac.activation import my_actFunc
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter
import numpy as np


class PinnNet(tn.Module):
    """


    """

    def __init__(self, indim=2, outdim=1, hidden_units=None, name2Model='DNN',
                 loss_function=torch.nn.MSELoss(reduction='mean'),
                 actName2in='tanh', actName='tanh', actName2out='linear',
                 ws=0.001, ds=0.0002):
        """
        :param indim: 输入的向量维度 本代码中默认为2维  时间和高度
        :param outdim: 输出的向量维度，本代码汇总默认设置为1未 浓度
        :param hidden_units: 隐藏层的数目
        :param name2Model: 使用的Model名字
        :param actName2in: 输入向量的激活函数
        :param actName: 隐藏层的激活函数
        :param actName2out: 输出层的激活函数
        :param ws\ds\p :pde的参数
        :param  loss_function:损失函数的格式
        """
        super(PinnNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()
        self.ws = ws
        self.ds = ds
        self.loss_function = loss_function

    def forward(self, inputs, scale=None, training=None, mask=None):
        """
        :param inputs: 输入的向量
        :param scale: 尚未理解
        :param training: 尚未理解
        :param mask: mask掉的数据
        :return:H 网络的输出（浓度）
        """
        # 输入的数据
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)
        H = self.actFunc_in(H)  # 对输入的数据进行一个激活处理

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer + 1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]
        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H

    def pde(self, inputs):
        u = self.forward(inputs)  # 神经网络得到的数据
        u_tx = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(self.forward(inputs)),
                                   create_graph=True, allow_unused=True)[0]  # 求偏导数
        d_t = u_tx[:, 0].unsqueeze(-1)  # 这里设置了t是第一维度
        d_x = u_tx[:, 1].unsqueeze(-1)  # 设置x是第二维度
        u_xx = torch.autograd.grad(d_x, inputs, grad_outputs=torch.ones_like(d_x),
                                   create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数

        #     ws=0.03 #Session 4.2
        #     ds=0.0001#table  1
        out_1 = d_t + self.ws * d_x - self.ds * u_xx
        out_2 = u
        return out_1, out_2

    def loss(self, inputs, labels):
        """
        :param inputs: 输入的向量
        :param labels:标签
        :param net: 网络
        :param Ws: PDE的系数
        :param Ds: PDE的系数
        :return: 1、方程与0的误差 2、预测值与实际值的误差
        """
        input1 = torch.cat([inputs[:, 0], inputs[:, 1]], 1)
        input2 = torch.cat([inputs[:, 2], inputs[:, 3]], 1)
        out_1, out_2 = self.pde(input1)
        out_3, out_4 = self.pde(input2)
        loss_interior_pde = self.loss_function(out_1, labels[:, 0])
        loss_interior_sol = self.loss_function(out_2, labels[:, 1])
        loss_boundary_pde = self.loss_function(out_3, labels[:, 2])
        loss_boundary_sol = self.loss_function(out_4, labels[:, 3])
        loss = loss_interior_pde + loss_interior_sol + loss_boundary_pde + loss_boundary_sol
        return loss


class DNN(tn.Module):
    def __init__(self, dim_in=2, dim_out=1, hidden_layers=None, name2Model='DNN', actName_in='tanh',
                 actName_hidden='tanh', actName_out='linear'):
        super(DNN, self).__init__()
        self.DNN = PinnNet(indim=dim_in, outdim=dim_out, hidden_units=hidden_layers, name2Model=name2Model,
                           actName2in=actName_in, actName=actName_hidden, actName2out=actName_out)

    def forward(self, x_input):
        out = self.DNN(x_input)
        return out

    def cal_l2loss(self, x_input, y_input):
        out = self.DNN(x_input)
        loss = torch.square(y_input - out)
        return loss

    def loss(self, inputs, labels):
        """
        :param inputs: 输入的向量
        :param labels:标签
        :param net: 网络
        :param Ws: PDE的系数
        :param Ds: PDE的系数
        :return: 1、方程与0的误差 2、预测值与实际值的误差
        """
        input1 = torch.cat([inputs[:, 0], inputs[:, 1]], 1)
        input2 = torch.cat([inputs[:, 2], inputs[:, 3]], 1)
        out_1, out_2 = self.pde(input1)
        out_3, out_4 = self.pde(input2)
        loss_interior_pde = self.loss_function(out_1, labels[:, 0])
        loss_interior_sol = self.loss_function(out_2, labels[:, 1])
        loss_boundary_pde = self.loss_function(out_3, labels[:, 2])
        loss_boundary_sol = self.loss_function(out_4, labels[:, 3])
        loss = loss_interior_pde + loss_interior_sol + loss_boundary_pde + loss_boundary_sol
        return loss

