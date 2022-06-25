#######################
# Datasets included in UCR15.

DATASETS_ma2021joint = [
    # \cite{ma2021joint}, several datasets from UCR15.
    # "we use single-label learning and joint-label learning to oversample four UCR datasets with different imbalance
    # ratios and test their performance. Here"
    'DistalPhalanxOutlineAgeGroup',
    'ECG5000',
    'Earthquakes',
    'ProximalPhalanxTW'
]

DATASETS_huang2019deep = [
    # FIXME: The current list just has 48 datasets because the primitive paper has two 'CBF' records and I cant not
    #  match the wrongly duplicated one to an other non-included dataset.
    # \cite{huang2019deep}, 49 datasets 49 datasets from UCR15.
    # "We evaluate all methods thoroughly on the UCR time series classification archive1, which consists of 49 datasets
    # selected from various real-world domains. "
    '50words',
    'Adiac',
    'ArrowHead',
    'Beef',
    'BeetleFly',
    'BirdChick',
    'CBF',
    'Car',
    'CinC_ECG_torso',
    'Coffee',
    'Cricket_X',
    'Cricket_Y',
    'Cricket_Z',
    'DiatomSizeReduction',
    'ECGFiveDays',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'Gun_Point',
    'Haptics',
    'Herring',
    'InlineSkate',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'Lighting2',
    'Lighting7',
    'Meat',
    'MedicalImages',
    'MoteStrain',
    'NonInvasiveFatalECG_Thorax1',
    'NonInvasiveFatalECG_Thorax2',
    'OSULeaf',
    'OliveOil',
    'Phoneme',
    'Plane',
    'ShapeletSim',
    'ShapesAll',
    'SonyAIBORobotSurface',
    'SonyAIBORobotSurfaceII',
    'SwedishLeaf',
    'Symbols',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'Wine',
    'WordsSynonyms',
    'Worms'
]
