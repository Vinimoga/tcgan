from ...lib.sktime import tsc_dataset_names as tsc_data

""" 128 UCR univariatetime series classification problems [1]"""
univariate = sorted(tsc_data.univariate)

""" 33 UEA multivariate time series classification problems [2]"""
multivariate = sorted(tsc_data.multivariate)

"""113 equal length/no missing univariate time series classification problems [3]"""
univariate_equal_length = sorted(tsc_data.univariate_equal_length)

"""11 variable length univariate time series classification problems [3]"""
univariate_variable_length = sorted(tsc_data.univariate_variable_length)

"""4 fixed length univariate time series classification problems with missing values"""
univariate_missing_values = sorted(tsc_data.univariate_missing_values)

"""26 equal length multivariate time series classification problems [4]"""
multivariate_equal_length = sorted(tsc_data.multivariate_equal_length)

"""7 variable length multivariate time series classification problems [4]"""
multivariate_unequal_length = sorted(tsc_data.multivariate_unequal_length)