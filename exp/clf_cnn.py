import os

from mlpy.lib.utils.path import makedirs, tag_path
from mlpy.tsc.expnn.run_ucr import run, MODELS_FOR_RUN
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets.ucr_uea.data_names import UCR85_DATASETS

from configure import DIR_LOG


def main(model_name):
    n_runs = 5
    data_name_list = ['50words']
    # data_name_list = UCR85_DATASETS
    fit_kwargs = dict(
        batch_size=16,
        n_epochs=300,
    )

    for i in range(n_runs):
        dir_log = os.path.join(DIR_LOG, tag, model_name, str(i))
        ModelClass, extend_dim = MODELS_FOR_RUN[model_name]
        makedirs(dir_log)
        run(ModelClass, data_name_list, DIR_DATA_UCR15, dir_log, extend_dim=extend_dim, fit_kwargs=fit_kwargs)
        print()
        print("Finish to run the model {}".format(model_name))
        print()


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))

    model_name = 'fcn'
    # model_name = 'resnet'
    main(model_name)

