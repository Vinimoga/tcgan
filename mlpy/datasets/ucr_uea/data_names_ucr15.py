"""
    There are some functions to categorize UCR15 datasets.
"""

import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

from mlpy.configure import DIR_DATASET, UCR15
from mlpy.datasets.ucr_uea.load import load_ucr, load_ucr_concat
from mlpy.lib.data import utils
from mlpy.lib.utils.path import get_dir_list

np.random.seed(123)

""" 79 datasets in which the test split is larger than the train split. last updated: 2018-12-15. """
test_larger_than_train = [
    '50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car',
    'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X',
    'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays',
    'Earthquakes', 'FISH', 'FaceAll', 'FaceFour', 'FacesUCR', 'FordA', 'FordB', 'Gun_Point',
    'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
    'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
    'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
    'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
    'OSULeaf', 'OliveOil', 'Phoneme', 'Plane', 'ProximalPhalanxTW', 'RefrigerationDevices',
    'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
    'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
    'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Two_Patterns',
    'UWaveGestureLibraryAll', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'synthetic_control',
    'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga']


def get_data_names_test_larger_than_train(data_root):
    data_name_list = get_dir_list(data_root)
    res = []
    for fname in data_name_list:
        datasets = load_ucr(fname, data_root)
        if datasets.test.X.shape[0] >= datasets.train.X.shape[0]:
            res.append(fname)
    return res


""" 45 datasets, (test set size) >= 2*(train set size) . last updated: 2018-12-15. """
test_double_than_train = [
    'ArrowHead', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
    'ECG5000', 'ECGFiveDays', 'Earthquakes', 'FaceAll', 'FaceFour', 'FacesUCR', 'FordA', 'FordB',
    'Gun_Point', 'HandOutlines', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'MALLAT',
    'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain',
    'Phoneme', 'ShapeletSim', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves',
    'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'TwoLeadECG', 'Two_Patterns',
    'UWaveGestureLibraryAll', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'uWaveGestureLibrary_X',
    'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga']


def get_data_names_test_double_than_train(data_root):
    data_name_list = get_dir_list(data_root)
    res = []
    for fname in data_name_list:
        datasets = load_ucr(fname, data_root)
        if datasets.test.X.shape[0] >= 2 * datasets.train.X.shape[0]:
            res.append(fname)
    return res


""" 33 datasets, (test set size) >= 2*(train set size) for each class. last updated: 2018-12-15. """
test_double_than_train_each_class = [
    'ArrowHead', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'DiatomSizeReduction', 'ECG5000',
    'ECGFiveDays', 'FacesUCR', 'FordA', 'FordB', 'Gun_Point', 'HandOutlines', 'InlineSkate',
    'InsectWingbeatSound', 'ItalyPowerDemand', 'MALLAT', 'MoteStrain', 'ShapeletSim',
    'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Symbols',
    'ToeSegmentation1', 'TwoLeadECG', 'Two_Patterns', 'UWaveGestureLibraryAll', 'Worms',
    'WormsTwoClass', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z',
    'wafer', 'yoga']


def get_data_names_test_double_than_train_each_class(data_root):
    data_name_list = get_dir_list(data_root)
    res = []
    for fname in data_name_list:
        datasets = load_ucr(fname, data_root)
        distr_train = utils.distribute_y(datasets.train.y)
        distr_test = utils.distribute_y(datasets.test.y)
        is_pass = True
        for key_tr in distr_train.keys():
            num_tr = distr_train[key_tr]
            num_te = distr_test[key_tr]
            if num_te is None or num_te < 2 * num_tr:
                is_pass = False
                break
        if is_pass:
            res.append(fname)
    return res


# ==============================================================================
#

def category_dataset_exact_length(dir_data):
    data_categories_length = {}
    fname_list = get_dir_list(dir_data)
    for fname in fname_list:
        data = load_ucr(fname, dir_data)
        length = data.train.X.shape[1]
        if (length in data_categories_length.keys()) is False:
            data_categories_length[length] = []
        data_categories_length[length].append(fname)
    for key in sorted(data_categories_length.keys()):
        print(key, data_categories_length[key])
    return data_categories_length


# ==============================================================================
#  category datasets get datasets name list
#  reference: http://timeseriesclassification.com/dataset.php
train_size = {
    '0to50': [  # 20 datasets
        'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'CinC_ECG_torso', 'Coffee', 'DiatomSizeReduction',
        'ECGFiveDays', 'FaceFour', 'Gun_Point', 'MoteStrain', 'OliveOil', 'ShapeletSim', 'SonyAIBORobotSurface',
        'SonyAIBORobotSurfaceII', 'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'TwoLeadECG'],
    '51to100': [  # 13 datasets
        'Car', 'ECG200', 'Herring', 'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat',
        'Trace', 'Wine', 'Worms', 'WormsTwoClass'],
    '101to500': [  # 36 datasets
        '50words', 'Adiac', 'ChlorineConcentration', 'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG5000', 'Earthquakes',
        'FISH', 'FacesUCR', 'Ham', 'HandOutlines', 'Haptics', 'InsectWingbeatSound', 'LargeKitchenAppliances',
        'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'OSULeaf',
        'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType',
        'SmallKitchenAppliances', 'Strawberry', 'SwedishLeaf', 'WordsSynonyms', 'synthetic_control', 'yoga'],
    '500toInf': [  # 16 datasets
        'ElectricDevices', 'FaceAll', 'FordA', 'FordB', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2',
        'PhalangesOutlinesCorrect', 'ProximalPhalanxOutlineCorrect', 'ShapesAll', 'StarLightCurves', 'Two_Patterns',
        'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer']
}


def category_dataset_trainsize(data_root):
    name_list = get_dir_list(data_root)

    names_0_to_50 = []
    names_51_to_100 = []
    names_101_to_500 = []
    names_greater_500 = []
    for fname in name_list:
        dataset = load_ucr(fname, data_root)
        n = dataset.train.X.shape[0]
        if n <= 50:
            names_0_to_50.append(fname)
        elif n <= 100:
            names_51_to_100.append(fname)
        elif n <= 500:
            names_101_to_500.append(fname)
        else:
            names_greater_500.append(fname)
    print("===== Category datasets by train size: ")
    print("form 0 to 50: ", len(names_0_to_50), names_0_to_50)
    print("from 51 to 100: ", len(names_51_to_100), names_51_to_100)
    print("from 101 to 500: ", len(names_101_to_500), names_101_to_500)
    print("greater than 500: ", len(names_greater_500), names_greater_500)
    print()


test_size = {
    '0to300': [  # 29 datasets
        'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'Coffee', 'Computers', 'ECG200', 'FISH', 'FaceFour',
        'Gun_Point', 'Ham', 'Herring', 'Lighting2', 'Lighting7', 'Meat', 'OSULeaf', 'OliveOil', 'Plane',
        'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ShapeletSim', 'ToeSegmentation1',
        'ToeSegmentation2', 'Trace', 'Wine', 'Worms', 'WormsTwoClass', 'synthetic_control'],
    '301to1000': [  # 32 datasets
        '50words', 'Adiac', 'CBF', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECGFiveDays', 'Earthquakes',
        'HandOutlines', 'Haptics', 'InlineSkate', 'LargeKitchenAppliances', 'MedicalImages',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'PhalangesOutlinesCorrect',
        'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapesAll', 'SmallKitchenAppliances',
        'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'Strawberry', 'SwedishLeaf', 'Symbols', 'WordsSynonyms'],
    '1001toInf': [  # 24 datasets
        'ChlorineConcentration', 'CinC_ECG_torso', 'ECG5000', 'ElectricDevices', 'FaceAll', 'FacesUCR', 'FordA',
        'FordB', 'InsectWingbeatSound', 'ItalyPowerDemand', 'MALLAT', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1',
        'NonInvasiveFatalECG_Thorax2', 'Phoneme', 'StarLightCurves', 'TwoLeadECG', 'Two_Patterns',
        'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer',
        'yoga']
}


def category_dataset_testsize(data_dir_root):
    name_list = get_dir_list(data_dir_root)

    names_0_to_300 = []
    names_301_to_1000 = []
    names_greater_1000 = []
    for fname in name_list:
        dataset = load_ucr(fname, data_dir_root)
        n = dataset.test.X.shape[0]
        if n <= 300:
            names_0_to_300.append(fname)
        elif n <= 1000:
            names_301_to_1000.append(fname)
        else:
            names_greater_1000.append(fname)
    print("===== Category datasets by test size: ")
    print("from 0 to 300: ", len(names_0_to_300), names_0_to_300)
    print("from 301 to 1000: ", len(names_301_to_1000), names_301_to_1000)
    print("greater than 1000: ", len(names_greater_1000), names_greater_1000)
    print()


length = {
    '0to300':  [  # 43 datasets
        '50words', 'Adiac', 'ArrowHead', 'CBF', 'ChlorineConcentration', 'Coffee', 'Cricket_X', 'Cricket_Y',
        'Cricket_Z', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200',
        'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FacesUCR', 'Gun_Point', 'InsectWingbeatSound',
        'ItalyPowerDemand', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW', 'MoteStrain', 'PhalangesOutlinesCorrect', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII',
        'Strawberry', 'SwedishLeaf', 'ToeSegmentation1', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'Wine', 'WordsSynonyms',
        'synthetic_control', 'wafer'],
    '301to700': [  # 25 datasets
        'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'DiatomSizeReduction', 'Earthquakes', 'FISH', 'FaceFour', 'FordA',
        'FordB', 'Ham', 'Herring', 'Lighting2', 'Lighting7', 'Meat', 'OSULeaf', 'OliveOil', 'ShapeletSim', 'ShapesAll',
        'Symbols', 'ToeSegmentation2', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z',
        'yoga'],
    '701toInf': [  # 17 datasets
        'CinC_ECG_torso', 'Computers', 'HandOutlines', 'Haptics', 'InlineSkate', 'LargeKitchenAppliances', 'MALLAT',
        'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'Phoneme', 'RefrigerationDevices', 'ScreenType',
        'SmallKitchenAppliances', 'StarLightCurves', 'UWaveGestureLibraryAll', 'Worms', 'WormsTwoClass']
}


def category_dataset_length(data_dir_root):
    name_list = get_dir_list(data_dir_root)

    names_0_to_300 = []
    names_301_to_700 = []
    names_greater_700 = []
    for fname in name_list:
        dataset = load_ucr(fname, data_dir_root)
        length = dataset.train.X.shape[1]
        if length <= 300:
            names_0_to_300.append(fname)
        elif length <= 700:
            names_301_to_700.append(fname)
        else:
            names_greater_700.append(fname)
    print("===== Category datasets by length: ")
    print("from 0 to 300: ", len(names_0_to_300), names_0_to_300)
    print("from 301 to 700: ", len(names_301_to_700), names_301_to_700)
    print("greater than 700: ", len(names_greater_700), names_greater_700)
    print()


nclasses = {
    '0to10': [  # 71 datasets
        'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'ChlorineConcentration', 'CinC_ECG_torso',
        'Coffee', 'Computers', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays', 'Earthquakes', 'ElectricDevices', 'FISH', 'FaceFour',
        'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'ItalyPowerDemand',
        'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 'OSULeaf',
        'OliveOil', 'PhalangesOutlinesCorrect', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
        'SmallKitchenAppliances', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry',
        'Symbols', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Two_Patterns',
        'UWaveGestureLibraryAll', 'Wine', 'Worms', 'WormsTwoClass', 'synthetic_control', 'uWaveGestureLibrary_X',
        'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'yoga'],
    '11to30': [  # 8 datasets
        'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'FaceAll', 'FacesUCR', 'InsectWingbeatSound', 'SwedishLeaf',
        'WordsSynonyms'],
    '31toInf': [  # 6 datasets
        '50words', 'Adiac', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'Phoneme', 'ShapesAll']
}


def category_dataset_nclass(data_dir_root):
    name_list = get_dir_list(data_dir_root)

    names_0_to_10 = []
    names_11_to_30 = []
    names_greater_30 = []
    for fname in name_list:
        dataset = load_ucr(fname, data_dir_root)
        n_class = dataset.nclass
        if n_class <= 10:
            names_0_to_10.append(fname)
        elif n_class <= 30:
            names_11_to_30.append(fname)
        else:
            names_greater_30.append(fname)
    print("===== Category datasets by the number of class: ")
    print("from 0 to 10: ", len(names_0_to_10), names_0_to_10)
    print("from 11 to 30: ", len(names_11_to_30), names_11_to_30)
    print("greater than 30: ", len(names_greater_30), names_greater_30)
    print()


DATA_ROOT = os.path.join(DIR_DATASET, UCR15)

if __name__ == '__main__':
    category_dataset_nclass(DATA_ROOT)
