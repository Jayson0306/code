import numpy as np
import matplotlib.pyplot as plt


def show_loss(loss_append):
    logloss = np.log(loss_append)
    plt.title("log(loss) trend")
    plt.plot(logloss, color='b', label='Label')
    plt.xlabel("iterations")
    plt.ylabel("log(loss)")
    plt.show()
