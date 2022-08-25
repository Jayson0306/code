import torch
import torch.nn as tn
import torch.nn.functional as tnf
import numpy as np
from torch.autograd import Variable
from activation import my_actFunc


class Pure_DenseNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        ws\ds\p :pde的参数
    """

    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32'):
        super(Pure_DenseNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0])
        tn.init.xavier_normal_(input_layer.weight)
        tn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1])
            tn.init.xavier_normal_(hidden_layer.weight)  # 参数初始化，这个也可以修改
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L2'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        return regular_w, regular_b

    def forward(self, inputs, scale=None, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)
        H = self.actFunc_in(H)

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


# ----------------Dense_ScaleNet(constructing NN and initializing weights and bias )------------
class Dense_ScaleNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(Dense_ScaleNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0])
        tn.init.xavier_normal_(input_layer.weight)
        tn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L2'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        return regular_w + regular_b

    def forward(self, inputs, scale=None, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)

        Unit_num = int(self.hidden_units[0] / len(scale))
        mixcoe = np.repeat(scale, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            torch_mixcoe = torch_mixcoe.cuda(device='cuda:'+str(self.gpu_no))
        H = self.actFunc_in(H*torch_mixcoe)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------dense net(constructing NN and initializing weights and bias )------------
class Dense_FourierNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', to_gpu=False, gpu_no=0,
                 repeat_Highfreq=True):
        super(Dense_FourierNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0], bias=False)
        tn.init.xavier_normal_(input_layer.weight)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            if i_layer == 0:
                hidden_layer = tn.Linear(in_features=2 * hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
            else:
                hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L2'):
        regular_w = 0.0
        regular_b = 0.0
        i_layer = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.abs(layer.bias))
                i_layer = i_layer + 1
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
                i_layer = i_layer + 1
        return regular_w + regular_b

    def forward(self, inputs, scale=None, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)

        Unit_num = int(self.hidden_units[0] / len(scale))
        mixcoe = np.repeat(scale, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            torch_mixcoe = torch_mixcoe.cuda(device='cuda:'+str(self.gpu_no))

        H = torch.cat([torch.cos(H*torch_mixcoe), torch.sin(H*torch_mixcoe)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# This is an example by means of the above module
class PDE_DNN(tn.Module):
    def __init__(self, dim_in=2, dim_out=1, hidden_layers=None, name2Model='DNN', actName_in='tanh',
                 actName_hidden='tanh', actName_out='linear', ws=0.001, ds=0.0002, use_gpu=False, no2gpu=0,
                 factor2freq=None, type2numeric='float32', opt2regular_WB='L2', HighFreq=False):
        super(PDE_DNN, self).__init__()
        if name2Model == 'DNN':
            self.DNN = Pure_DenseNet(indim=dim_in, outdim=dim_out, hidden_units=hidden_layers, name2Model=name2Model,
                                     actName2in=actName_in, actName=actName_hidden, actName2out=actName_out,
                                     type2float=type2numeric)
        elif name2Model == 'Scale_DNN':
            self.DNN = Dense_ScaleNet(indim=dim_in, outdim=dim_out, hidden_units=hidden_layers, name2Model=name2Model,
                                      actName2in=actName_in, actName=actName_hidden, actName2out=actName_out,
                                      to_gpu=use_gpu, gpu_no=no2gpu, type2float=type2numeric,
                                      repeat_Highfreq=HighFreq)
        elif name2Model == 'Fourier_DNN':
            self.DNN = Dense_FourierNet(indim=dim_in, outdim=dim_out, hidden_units=hidden_layers, name2Model=name2Model,
                                        actName2in=actName_in, actName=actName_hidden, actName2out=actName_out,
                                        to_gpu=use_gpu, gpu_no=no2gpu, type2float=type2numeric,
                                        repeat_Highfreq=HighFreq)
        self.ws = ws
        self.ds = ds

        self.factor2freq = factor2freq
        self.opt2regular_WB = opt2regular_WB

    def evaluate(self, x_input):
        UNN = self.DNN(x_input, scale=self.factor2freq)
        return UNN

    def loss_it_pde(self, inputs, labels):
        """
        :param inputs: 输入的向量
        :param labels:标签
        :param net: 网络
        :param Ws: PDE的系数
        :param Ds: PDE的系数
        :return: 1、pde方程的内部点误差
        """

        UNN = self.DNN(inputs, scale=self.factor2freq)  # 神经网络得到的数据
        u_tx = torch.autograd.grad(UNN, inputs, grad_outputs=torch.ones_like(self.evaluate(inputs)),
                                   create_graph=True, allow_unused=True)[0]  # 求偏导数
        d_t = u_tx[:, 0].unsqueeze(-1)  # 这里设置了t是第一维度
        d_x = u_tx[:, 1].unsqueeze(-1)  # 设置x是第二维度
        u_xx = torch.autograd.grad(d_x, inputs, grad_outputs=torch.ones_like(d_x),
                                   create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数

        #     ws=0.03 #Session 4.2
        #     ds=0.0001#table  1
        loss_pde = d_t + self.ws * d_x - self.ds * u_xx
        # 生成pde的损失
        all_zeros = np.zeros(labels.size())
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False)
        loss_it_pde = loss_function(loss_pde, pt_all_zeros)
        return loss_it_pde

    # def loss_bd_pde(self, inputs, labels):
    #     UNN = self.DNN(inputs, scale=self.factor2freq)  # 神经网络得到的数据
    #     u_tx = torch.autograd.grad(UNN, inputs, grad_outputs=torch.ones_like(self.evaluate(inputs)),
    #                                create_graph=True, allow_unused=True)[0]  # 求偏导数
    #     d_t = u_tx[:, 0].unsqueeze(-1)  # 这里设置了t是第一维度
    #     d_x = u_tx[:, 1].unsqueeze(-1)  # 设置x是第二维度
    #     u_xx = torch.autograd.grad(d_x, inputs, grad_outputs=torch.ones_like(d_x),
    #                                create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数
    #
    #     #     ws=0.03 #Session 4.2
    #     #     ds=0.0001#table  1
    #     loss_pde = d_t + self.ws * d_x - self.ds * u_xx
    #     # 生成pde的损失
    #     all_zeros = np.zeros(labels.size())
    #     pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False)
    #     loss_bd_pde = loss_function(loss_pde, pt_all_zeros)
    #     return loss_bd_pde

    def loss_it_net(self, inputs, labels):
        UNN = self.DNN(inputs, scale=self.factor2freq)
        # loss_net = torch.nn.MSELoss()(UNN, labels)
        loss_net = loss_function(UNN, labels)
        return loss_net
        # 神经网络得到的数据

    def loss_bd_net(self, inputs, labels):
        UNN = self.DNN(inputs, scale=self.factor2freq)
        loss_net = loss_function(UNN, labels)
        return loss_net

    def loss_init_net(self, inputs, labels):
        UNN = self.DNN(inputs, scale=self.factor2freq)
        loss = loss_function(UNN, labels)
        return loss

    def loss_it_pde_dt(self, inputs, labels):
        """
        :param inputs: 输入的向量
        :param labels:标签
        :param net: 网络
        :param Ws: PDE的系数
        :param Ds: PDE的系数
        :return: 1、pde方程的内部点误差
        """

        UNN = self.DNN(inputs, scale=self.factor2freq)  # 神经网络得到的数据
        u_tx = torch.autograd.grad(UNN, inputs, grad_outputs=torch.ones_like(self.evaluate(inputs)),
                                   create_graph=True, allow_unused=True)[0]  # 求偏导数
        d_t = u_tx[:, 0].unsqueeze(-1)  # 这里设置了t是第一维度
        d_x = u_tx[:, 1].unsqueeze(-1)  # 设置x是第二维度
        u_xx = torch.autograd.grad(d_x, inputs, grad_outputs=torch.ones_like(d_x),
                                   create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数

        #     ws=0.03 #Session 4.2
        #     ds=0.0001#table  1
        loss_pde = d_t + self.ws * d_x - self.ds_function(inputs[:, 1]) * u_xx
        # 生成pde的损失
        all_zeros = np.zeros(labels.size())
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False)
        loss_it_pde = loss_function(loss_pde, pt_all_zeros)
        return loss_it_pde


def loss_function(inputs, labels, loss_name='MSE'):
    if loss_name == 'MSE':
        loss_out = torch.nn.MSELoss(reduction='sum')(inputs, labels)
    elif loss_name == 'BCE':
        loss_out = torch.nn.BCELoss()(inputs, labels)
    return loss_out


def dt(z):
    return 1 / z
