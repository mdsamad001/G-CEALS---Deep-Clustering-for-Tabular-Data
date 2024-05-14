from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from utils import metrics, tabular_data
from utils.openml import get_data
import numpy as np
import time

def get_dataset_info(X_actual, y_actual):
    return X_actual.shape[0], X_actual.shape[1], len(np.unique(y_actual))


def do_kmeans(dataset, n_classes=0):
    X_actual, y_actual = get_data(dataset)

    n_classes = len(np.unique(y_actual)) if n_classes == 0 else n_classes

    # Standardize data
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X_actual)

    kmeans = KMeans(n_clusters=n_classes, init="k-means++", n_init='auto', random_state=42)
    start_time = time.time()
    y_pred = kmeans.fit_predict(X)
    time_taken = time.time() - start_time

    from utils.pickle import save_var
    save_var(y_pred, f'./predictions/kmeans-x_{dataset}.pkl')

    acc = metrics.acc(y_actual, y_pred)
    nmi = metrics.nmi(y_actual, y_pred)
    ari = metrics.ari(y_actual, y_pred)

    info = get_dataset_info(X_actual, y_actual)
    return [*info, n_classes, time_taken, acc, nmi, ari]


def do_gmm(dataset, n_classes=0):
    X_actual, y_actual = get_data(dataset)

    n_classes = len(np.unique(y_actual)) if n_classes == 0 else n_classes

    # Standardize data
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X_actual)

    gmm = GaussianMixture(n_components=n_classes, init_params="kmeans", random_state=42)
    start_time = time.time()
    y_pred = gmm.fit_predict(X)
    time_taken = time.time() - start_time

    from utils.pickle import save_var
    save_var(y_pred, f'./predictions/gmm-x_{dataset}.pkl')
    
    acc = metrics.acc(y_actual, y_pred)
    nmi = metrics.nmi(y_actual, y_pred)
    ari = metrics.ari(y_actual, y_pred)

    info = get_dataset_info(X_actual, y_actual)
    return [*info, n_classes, time_taken, acc, nmi, ari]