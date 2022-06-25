# TCGAN: Convolutional Generative Adversarial Network for Time Series Classification and Clustering

## Set up environment
- the code is usually ran on Mac OS and Ubuntu 20.04.3 LTS.
- clone the project and `cd tcgan`
- create a new environment:
  - `conda create --name tcgan python=3.6`
  - `conda activate tcgan`
  - `pip install -r requirements.txt`
    - Note: if you have GPU, please replace `tensorflow==2.3.0` with `tensorflow-gpu 2.3.0` first.
- add the project path, the sources root is `tcgan`
  - `export PYTHONPATH="$PYTHONPATH:$PWD"`
  - double check `echo $PYTHONPATH`
- train a TCGAN `python exp/tcgan_.py`, you will see the results below `cache/exp_tcgan_` directory.
- `run.sh` is for multiple runs.

## Project structure

- `cache/`: the directory to store outputs.
- `exp/`: scripts for experiments.
- `mlpy/`: common lib.
- `raw-data/`: raw data. There is a sample dataset `50words`, you can get other datasets from [UCR Archive](www.cs.ucr.edu/~eamonn/time_series_data/). 
- `tcgan/`: implementations of models.
  - `lib/`
  - `model/`
- `run.sh`: a shell script for multiple runs.

## Experiments

All scripts for main experiments are saved below `exp/`

1. You can train a TCGAN with `tcgan_.py`. Similarly, you can also train the benchmarks with `ae.py`, `ae_ae.py`, and `ae_vae.py`.
2. The pre-trained models are reused for general classification in `clf_tcgan.py`, `clf_ae_ae.py`, `clf_ae_dae.py` and `clf_ae_vae.py`. Purely supervised baselines are `clf_rawdata.py` and `clf_cnn.py`. In addition, `tcgan_d.py` trains the discriminator of TCGAN from scratch for multi-classification. `tcgan_d_random.py` is a randomly initialized discriminator
3. `clf_imb_tcgan.py` is used for the imbalanced classification.
4. `semisupervised_cnns.py` and `semisupervised_tcgan.py` investigate the influence of the size of labeled training dataset.
5. `clus_rawdata.py`, `clus_kshape.py` and `clus_tcgan.py` are implemented for clustering.
6. `vis.py`, `vis_rawdata.py`, `vis_tcgan.py` are used to inspect the representation space.

Other open source benchmarks are [TimeGAN](https://github.com/jsyoon0823/TimeGAN) and [CotGAN](https://github.com/tianlinxu312/cot-gan).
