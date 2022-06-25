import numpy as np


def vote_majority(y_pred, batch_size):
    y_pred_vote = []
    n_bach = int(y_pred.shape[0] / batch_size)
    for i in range(n_bach):
        unique_val, sub_ind, correspond_ind, count = np.unique(
            y_pred[i*batch_size:(i+1)*batch_size], True, True, True)
        idx_max = np.argmax(count)
        label = unique_val[idx_max]
        y_pred_vote.append(label)

    return y_pred_vote