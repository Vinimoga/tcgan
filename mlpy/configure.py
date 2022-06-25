import os
import mlpy


DIR_ROOT = os.path.dirname(os.path.dirname(mlpy.__file__))
DIR_DATASET = os.path.join(DIR_ROOT, 'raw-data')
TAG_CACHE = 'cache'

"""Some datasets"""

UCR15 = 'UCR_TS_Archive_2015'
UCR18 = 'Univariate_ts'
UEA_MTS = 'Multivariate_ts'

DIR_DATA_UCR15 = os.path.join(DIR_DATASET, UCR15)
DIR_DATA_UEA_MTS = os.path.join(DIR_DATASET, UEA_MTS)


