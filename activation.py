import torch
import torch.nn as tn
import torch.nn.functional as tnf
import numpy as np


# optional activation function 定义了可以选择的激活函数
class my_actFunc(tn.Module):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName

    def forward(self, x_input):
        if str.lower(self.actName) == 'relu':
            out_x = tnf.relu(x_input)
        elif str.lower(self.actName) == 'leaky_relu':
            out_x = tnf.leaky_relu(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = torch.tanh(x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tnf.relu(x_input) * tnf.relu(1 - x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tnf.relu(x_input) * tnf.relu(1 - x_input) * torch.sin(2 * np.pi * x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tnf.elu(x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = torch.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5 * torch.sin(x_input) + 0.5 * torch.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tnf.sigmoid(x_input)
        elif str.lower(self.actName) == 'gelu':
            out_x = tnf.gelu(x_input)
        else:
            out_x = x_input
        return out_x
