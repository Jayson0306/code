from net2 import *
from data_gen2 import DataGen2
from graphing import plot_heatmap
from data_loss import *
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == "__main__":
    starttime = datetime.now()
    # 输出数据的初始化
    zs = 0.1
    ze = 2
    ts = 1
    te = 3000
    zsteps = 10
    tsteps = 300
    it_batch_size = 2000
    bd_batch_size = 500
    init_batch_size = 500
    ws = 0.001
    ds = 0.0002
    p = 0.0001
    # 生成数据

    data = DataGen2(zs, ze, ts, te, zsteps=zsteps, tsteps=tsteps, ws=ws, ds=ds, p=p)
    dim_in = 2
    dim_out = 1
    # hidden_list = (50, 40, 30, 30, 20)
    hidden_list = (20, 10, 8, 8, 5)
    model_name = 'DNN'
    init_lr = 0.01
    max_it = 1000

    # 神经网络各组分的初始化
    # model = DNN1(indim=dim_in, outdim=dim_out, hidden_units=hidden_list, name2Model=model_name, actName2in='tanh',
    #                 actName='tanh')
    # model = Pure_DenseNet(indim=dim_in, outdim=dim_out, hidden_units=hidden_list, name2Model=model_name, actName2in='tanh',
    #                 actName='tanh')
    # model = PDE_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layers=hidden_list, name2Model=model_name, actName_in='tanh',
    # actName_hidden='tanh', ws=ws, ds=ds)
    model = PDE_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layers=hidden_list, name2Model=model_name, actName_in='sin',
                    actName_hidden='sin', ws=ws, ds=ds)
    params2Net = model.DNN1.parameters()
    optimizer = torch.optim.Adam(params2Net, lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    arr2epoch = []
    arr2loss = []
    arr2lr = []
    predi = []
    loss_append = []
    lr_append = []
    init_netloss = []
    it_netloss = []
    bd_netloss = []
    for i_epoch in range(max_it):

        # 每次迭代都随机生成数据
        it_inputs, it_labels = data.gen_inter_m(it_batch_size)
        bd_inputs, bd_labels = data.gen_bound(bd_batch_size)
        init_inputs, init_labels = data.gen_init(init_batch_size)
        loss1 = model.loss_init_net(init_inputs, init_labels)
        loss2 = model.loss_it_pde(it_inputs, it_labels)
        loss3 = model.loss_bd_net(bd_inputs, bd_labels)
        # ds是常数损失函数修改了 内部点的pde损失
        # loss = 1000 * model.loss_it_net(it_inputs, it_labels) + model.loss_it_pde(it_inputs, it_labels) \
        # + 1000 * model.loss_bd_net(bd_inputs, bd_labels)
        loss = 100 * loss1 + loss2 + 200*loss3
        # loss = 1000 * model.loss_init_net(init_inputs, init_labels) + model.loss_it_pde(it_inputs, it_labels) \
        # + 1000 * model.loss_bd_net(bd_inputs, bd_labels)
        loss_append.append(loss.item())
        init_netloss.append(loss1.detach().numpy())
        it_netloss.append(loss2.detach().numpy())
        bd_netloss.append(loss3.detach().numpy())
        lr_append.append(optimizer.param_groups[0]['lr'])

        prediction = model.evaluate(it_inputs)
        predi.append(prediction)
        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 求偏导
        optimizer.step()  # 更新参数
        scheduler.step()

        if i_epoch % 100 == 0:
            print('i_epoch --- loss:', i_epoch, loss.item())
            # print("第%d个epoch的学习率：%f" % (i_epoch, optimizer.param_groups[0]['lr']))
            arr2loss.append(loss.item())
            arr2lr.append(optimizer.param_groups[0]['lr'])
    endtime = datetime.now()
    print("RunTime: {}h-{}m-{}s".format(endtime.hour - starttime.hour, endtime.minute - starttime.minute,
                                        endtime.second - starttime.second))
    # -------------show loss----------#
    logloss = np.log(loss_append)
    plt.title("log(sum loss) trend")
    plt.plot(logloss, color='b', label='Label')
    plt.xlabel("iterations")
    plt.ylabel("log(loss)")
    plt.show()
    plot_heatmap(data, model, max_it)
    log_0 = np.log(it_netloss)
    log_1 = np.log(bd_netloss)
    log_2 = np.log(init_netloss)
    log_3 = np.log(loss_append)
    plt.title("loss trend")
    plt.plot(log_0, color='r')
    plt.plot(log_1, color='b')
    plt.plot(log_2, color='y')
    plt.plot(log_3, color='g')
    plt.legend(('interior_net loss', 'bd_net loss', 'init_net loss', 'total loss'))
    plt.savefig("D:\\photosaved\\loss_comprt_iterations%d_networksizeSmall.png" % max_it)
