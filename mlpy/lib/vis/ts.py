import os
import shutil
import matplotlib.pyplot as plt


def plot_data_by_class(data_dict, out_dir):
    """ TODO: This function hasn't been used for a long time. Please test it first.
    ---------------
    example:

    def plot_data_by_class_batch(data_dir_root, data_name_list, out_dir='./cache/vis'):
        from mlpy.datasets.ucr_uea import load_ucr_concat
        from mlpy.lib.data.utils import distribute_dataset
        for data_name in data_name_list:
            print("preprocessing datasets: {}".format(data_name))
            X_all, y_all = load_ucr_concat(data_name, data_dir_root)
            data_dict = distribute_dataset(X_all, y_all)
            out_dir_current = os.path.join(out_dir, data_name)
            plot_data_by_class(data_dict, out_dir_current)
    ---------------

    """
    # make directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # plot and save figure
    for key in data_dict.keys():
        samples = data_dict[key]
        n_sample = samples.shape[0]
        title = 'class-{}_nsample-{}'.format(key, n_sample)
        fname = title + '.png'
        n_plot = min(n_sample, 4)
        samples_plot = samples[:n_plot]
        f, axes = plt.subplots(n_plot // 2, 2)
        axes = axes.flat[:]
        f.suptitle(title)
        for i, ax in enumerate(axes):
            if i >= samples_plot.shape[0]:
                break
            ax.plot(samples_plot[i])
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()



