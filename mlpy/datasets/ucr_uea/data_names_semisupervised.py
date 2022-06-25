#######################
# Datasets included in UCR15.

DATASETS_jawed2020self = [
    # 13 datasets from the paper: Self-supervised learning for semi-supervised time series classification
    'Coffee',
    'CBF',
    'ECG200',
    'FaceFour',
    'OSULeaf',
    'ItalyPowerDemand',
    'Lighting2',
    'Lighting7',
    'Gun_Point',
    'Trace',
    'WordsSynonyms',
    'OliveOil',
    'StarLightCurves'
]

DATASETS_fan2021semi = [
    # this paper select 3 relatively large datasets.
    # paper: Semi-Supervised Time Series Classification by Temporal Relation Prediction.
    'Cricket_X',
    'InsectWingbeatSound',
    'UWaveGestureLibraryAll'
]

DATASETS_goschenhofer2021deep = [
    # this paper select large datasets.
    # "We select three of the largest datasets from this repository, namely Crop, ElectricDevices and FordB."
    # paper: Deep Semi-Supervised Learning for Time Series Classification
    'Crop',  # Note: this dataset is in the UCR18/UCR128 archive instead of UCR15.
    'ElectricDevices',
    'FordB'
]

DATASETS_goschenhofer2021deep_UCR15 = [
    'ElectricDevices',
    'FordB'
]


DATASETS_SSL_UCR15 = list(set(DATASETS_jawed2020self + DATASETS_fan2021semi + DATASETS_goschenhofer2021deep_UCR15))



