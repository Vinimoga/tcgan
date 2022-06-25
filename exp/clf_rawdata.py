import os
import fire
import json
import pandas as pd

from mlpy.lib.utils.path import tag_path, makedirs
from mlpy.lib.utils.log import set_logging
from mlpy.configure import DIR_DATA_UCR15
from mlpy.datasets import ucr_uea

from configure import DIR_LOG
from tcgan.lib.classfiers import *


class ClfUCR(object):
    def __init__(self,
                 tag,
                 log_dir,
                 data_dir,
                 data_name,
                 **kwargs):
        self.tag = tag
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.data_name = data_name
        self.kwargs = kwargs

        self.logger = set_logging("{}_{}".format(self.tag, self.data_name), self.log_dir)

        self._init_raw_data()

    def _init_raw_data(self):
        x_tr, y_tr, x_te, y_te, n_classes = ucr_uea.load_ucr(self.data_name, self.data_dir, one_hot=True)

        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.n_classes = n_classes
        self.input_shape = self.x_tr.shape[1:]

    def run(self):
        clfs = self.kwargs['classifiers']
        res = {}
        for name, clf in clfs.items():
            print(f"processing {name}")
            _res = clf(self.x_tr, self.y_tr, self.x_te, self.y_te, self.n_classes)
            for k, v in _res.items():
                res[f'{name}_{k}'] = v

        out_file = self.kwargs['out_file']
        with open(os.path.join(self.log_dir, out_file), 'a') as f:
            f.write(json.dumps(res) + "\n")


class Experiment(object):
    def __init__(self, tag, data_dir, data_name_list, log_dir, exp_uni_obj=None, **kwargs):
        self.tag = tag
        self.data_dir = data_dir
        self.data_name_list = data_name_list
        self.log_dir = log_dir
        self.exp_uni_obj = exp_uni_obj
        if self.exp_uni_obj is None:
            self.exp_uni_obj = ClfUCR
        self.kwargs = kwargs

    def run(self, i_run=0):
        log_dir = makedirs(os.path.join(self.log_dir, str(i_run)))
        logger = set_logging(self.tag, log_dir)

        for data_name in self.data_name_list:
            logger.info(f"****** process dataset {data_name}")
            _log_dir = makedirs(os.path.join(log_dir, data_name))

            exp_unit = self.exp_uni_obj(self.tag, _log_dir, self.data_dir, data_name, **self.kwargs)
            exp_unit.run()

        self.reduce(log_dir, self.data_name_list, [self.kwargs['out_file']], self.kwargs['res_out_fname'])

    @staticmethod
    def reduce(log_dir, data_name_list, json_fnames, out_fname):
        res_list = []
        for data_name in data_name_list:
            res = {'data_name': data_name}
            for fname in json_fnames:
                with open(os.path.join(log_dir, data_name, fname), 'r') as f:
                    last = f.readlines()[-1]
                    res.update(json.loads(last))
            res_list.append(res)

        df = {}
        if len(res_list) > 0:
            for key in res_list[0].keys():
                df[key] = []
            for r in res_list:
                for key in r.keys():
                    df[key].append(r[key])
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(log_dir, f'{out_fname}.csv'), index=False)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    log_dir = makedirs(os.path.join(DIR_LOG, tag))
    logger = set_logging(tag, log_dir)
    data_name_list = ['50words']
    # data_name_list = UCR85_DATASETS

    exp_tag = 'raw-ts'
    exp_cfg = {
        'classifiers': {
            'softmax': softmax,
            'knn': knn,
            'lr': lr,
            'lsvc': lsvc,
            'svc': svc
        },
        'out_file': 'clf_{}.json'.format(exp_tag),
        'res_out_fname': 'res_clf_{}'.format(exp_tag)
    }

    exp = Experiment(tag, DIR_DATA_UCR15, data_name_list, log_dir, **exp_cfg)

    fire.Fire(exp.run)


