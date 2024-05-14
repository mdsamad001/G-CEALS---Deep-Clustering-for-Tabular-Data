from sklearn import metrics
import numpy as np
from functools import partial

ari = metrics.adjusted_rand_score
nmi = partial(metrics.normalized_mutual_info_score, average_method="arithmetic")
# nmi = metrics.adjusted_mutual_info_score

# def acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed

#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`

#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     # from sklearn.utils.linear_assignment_ import linear_assignment # deprecated
#     from scipy.optimize import linear_sum_assignment as linear_assignment
#     ind = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# pasted from gceals
# default code can not handle when N(y_pred) != N(y_true)
def linear_assignment(cost_matrix):
      try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
      except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
      
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size