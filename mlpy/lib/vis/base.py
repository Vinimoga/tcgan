import matplotlib.pyplot as plt
import os


def plot_loss(loss, loss_val=None, dir_save=None, img_name=None, title=None):
    plt.plot(loss, label='training')
    if loss_val is not None:
        plt.plot(loss_val, label='validation')
        plt.legend(loc='best')
    if title is not None:
        plt.title(title)
    plt.ylim([0, max(plt.ylim())])
    if dir_save is not None:
        plt.savefig(os.path.join(dir_save, '{}.png'.format('loss' if (img_name is None) else img_name)))


def plot_acc(acc, acc_val=None, dir_save=None, img_name=None, title=None):
    plt.plot(acc, label='training')
    if acc_val is None:
        plt.plot(acc_val, label='validation')
        plt.legend(loc='best')
    if title is not None:
        plt.title(title)
    plt.ylim([0, 1])
    if dir_save is not None:
        plt.savefig(os.path.join(dir_save, '{}.png'.format('acc' if (img_name is None) else img_name)))