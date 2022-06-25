import numpy as np

from statsmodels.tsa.stattools import acf


def autocorr(x, n_lags=10, return_dict=True):
    n_samples, length = x.shape
    assert length > n_lags, f"n_lags={n_lags} should be less than the time series length {length}."
    avg_ac = np.zeros(n_lags)
    for i in range(n_samples):
        ts = x[i]
        ac = acf(ts, fft=False, nlags=n_lags)
        if np.sum(np.isnan(ac)) > 0:
            continue
        avg_ac = avg_ac + ac[1:(1+n_lags)]

    avg_ac /= n_samples

    if return_dict:
        res = {}
        for i, val in enumerate(avg_ac):
            res[f"lag-{i+1}"] = val
    else:
        res = avg_ac

    return res


def znorm_statistics(x, return_dict=True):
    mean = np.round(np.mean(np.mean(x, axis=1)), 2)
    std = np.round(np.mean(np.std(x, axis=1)), 2)

    if return_dict:
        res = {'mean': mean, 'std': std}
    else:
        res = (mean, std)

    return res
