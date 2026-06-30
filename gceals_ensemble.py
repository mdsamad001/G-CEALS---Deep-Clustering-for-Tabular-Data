from functools import partial
import time
from utils import metrics
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

from utils import metrics
from utils.debug import *
from utils.dataloader import EqualLoader

import pandas as pd
import pickle
import os

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    print("CUDA available")


# Network
# x-500-500-2000-10=2000-500-500=x_hat;

class Autoencoder(nn.Module):
    def __init__(self, input_level, output_level=10):
        super().__init__()
        level_1 = 500
        level_2 = 500
        level_3 = 2000
        self.output_level = output_level

        activation = nn.ReLU
        # activation = partial(nn.LeakyReLU, 0.2)

        # Encoder, decoder blocks
        self.encoder = nn.Sequential(
            nn.Linear(input_level, level_1),
            activation(),
            nn.Linear(level_1, level_2),
            activation(),
            nn.Linear(level_2, level_3),
            activation(),
            nn.Linear(level_3, output_level),
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_level, level_3),
            activation(),
            nn.Linear(level_3, level_2),
            activation(),
            nn.Linear(level_2, level_1),
            activation(),
            nn.Linear(level_1, input_level),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

class Clustering(nn.Module):
    def __init__(self, pretrained_autoencoder, n_clusters,
                 init_centriods=None, init_covs=None, args=None):
        super().__init__()
        self.ae = pretrained_autoencoder
        self.n_clusters = n_clusters
        self.args = args

        # Initialize learnable parameter cluster centroids
        self.centroids = nn.Parameter(torch.randn(
            self.n_clusters, self.ae.output_level, dtype=torch.float
        ) if init_centriods == None else init_centriods)

        # Initialize learnable parameter covariance
        # Init K identity matrices
        # parenthesis to break long statements
        init_covs = (torch.eye(self.ae.output_level, dtype=torch.float)
                     .unsqueeze(dim=0).repeat(n_clusters, 1, 1)
                     if init_covs == None else init_covs)


        self.identity_var = torch.eye(self.ae.output_level, dtype=torch.float, device=args.device)
        diag = torch.arange(self.ae.output_level)
        init_covs_diag = init_covs[:, diag, diag].reshape(-1, 1, diag.shape[0])
        # softplus(x) = 1/B log_e (1 + exp(B*x))
        softplus_reverse = torch.log(torch.exp(init_covs_diag * 1) - 1)

        # Cov = I * softplus(add_var)
        self.add_var = nn.Parameter(softplus_reverse)

        activation = nn.ReLU

        """
        This SOFTMAX HEAD contributes to the Q distribution from the Autoencoder Z space
        ae.output_level = dimension of the Z or latent space of the autoencoder.
        """
        self.softmax_head = nn.Sequential(
            nn.Linear(self.ae.output_level, self.ae.output_level),
            activation(),
            nn.Linear(self.ae.output_level, self.n_clusters),
        )
    def get_cov(self):
        # Cov = I * softplus(add_var)
        sigma = self.identity_var*F.softplus(self.add_var)
        return sigma

    def get_gceals_target(self, z):

        '''Compute Target Q using Softmax of Distance between Centroid and Data Points'''

        x = z.unsqueeze(dim=1)  # (n, d) -> (n, 1, d)
        centroids = self.centroids
        mu = centroids.unsqueeze(dim=0)  # (k, d) -> (1, k, d)

        sigma_inv = torch.pinverse(self.get_cov())

        d = x - mu  # (n, 1, d) - (1, k, d) -> (n, k, d)
        d1 = d.unsqueeze(dim=2)  # (n, k, d) -> (n, k, 1, d)
        # (n, k, 1, d) \times (n, k, d, d) -> (n, k, 1, d)
        d2 = d1.matmul(sigma_inv)
        # (n, k, 1, d) \times (n, k, d, 1) -> (n, k, 1, 1) -> (n, k, 1)
        d3 = d2.matmul(d1.transpose(2, 3)).squeeze()
        # S = softmax of mahalanobis distances
        d_final = d3.pow(0.5)

        softmax_p = F.softmax(-d_final, dim=1)

        return softmax_p

    def softmax_output(self, z):
        q = self.softmax_head(z)
        return q

    def cluster_output(self, z):
        p = self.get_gceals_target(z)
        return p

    def forward(self, x):
        z, x_hat = self.ae(x)

        """
        P = Target, Q = Source Distributions
        """
        p = self.cluster_output(z)
        q = self.softmax_output(z)

        return p, x_hat, q


'''
--> pretain () is used to pretain the autoencoder (standalone), optimizing the reconstruction loss only - inspired by IDEC
--> train (): pretained Autoencoder is used in a joint AE + CE cluster training using train ()

'''

def pretrain(data_tensor, args):

    # Initialize model
    ae_model = Autoencoder(input_level=data_tensor.size()[1],
                           output_level=args.latent_dim).to(args.device)


    file_name = f'pretrained-ae_dim-{args.latent_dim}_epoch-{args.pretrain_epochs}_{args.dataset}'
    ae_model_path = f'./models/{file_name}.pth'
    ae_model_state = False

    # ugly try catch if-like statement to skip pretraining if model is saved
    try:

        ae_model_state = torch.load(ae_model_path)
        ae_model.load_state_dict(ae_model_state)
        print("Using saved pretrained ae_model")

        if not args.redo_pretrain:
            return ae_model
    except Exception as e:
        print("Pretraining from scratch")
        # print(e)

    # loss and log
    criterion = nn.MSELoss()
    ae_loss_list = []  # reconstruction loss


    # prepare for pretraining
    lr = args.l_rate  # default l_rate 1e3
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=lr)
    pretrain_loader = DataLoader(
        data_tensor, batch_size=args.batch_size, shuffle=True)


    # Pretrain ae_model
    ae_model.train()
    pbar = tqdm(range(args.pretrain_epochs))
    for epoch in pbar:
        for batch_x in pretrain_loader:
            batch_z, x_hat = ae_model(batch_x)

            ae_loss = criterion(x_hat, batch_x)

            loss = ae_loss

             # These three lines are needed to train a NN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ae_loss_list.append(ae_loss.item())

            pbar.set_description((
                f'Pretraining | ae_loss: {ae_loss.item():.5f};'
            ))

    os.makedirs(os.path.dirname(ae_model_path), exist_ok=True)
    if not args.redo_pretrain:
        torch.save(ae_model.state_dict(), ae_model_path)
        print("Saved pretrained ae_model to save time later on")

    # plot pretraining loss curve
    plt.figure(figsize=(10, 7))
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.plot(range(len(ae_loss_list)), ae_loss_list, 'r',
             label="Reconstruction loss, $L_{recon}$")
    # plt.plot(range(len(kl_loss_list)),kl_loss_list, 'g', label="BCE_loss, $L_{bce}$")
    plt.legend()
    plt.title('Pretraining')
    plt.savefig(f'./debug/{file_name}_loss.png', bbox_inches='tight')
    plt.close()

    return ae_model


'''
    train() takes a pretrained autoencoder and finetunes it using a joint loss = ae_loss + ce_loss

'''

def train(ae_model, data_tensor, y_actual, args):
    method = 'gceals'
    file_name = f'{method}_dim-{args.latent_dim}_epoch-{args.finetune_epochs}_gamma={args.gamma}_{args.dataset}'

    data_numpy = data_tensor.cpu().numpy()

    # get Z from pretrained AE
    ae_model.eval()
    with torch.no_grad():
        Z_tensor, _ = ae_model(data_tensor)
        Z = Z_tensor.cpu().numpy()

    # initialize centroids with Z
    init_centroids = None
    init_covs = None


    kmeans = KMeans(n_clusters=args.n_clusters,
                    init="k-means++", n_init='auto', random_state=42)
    y_pred_last = kmeans.fit_predict(Z)
    kmeans_y = torch.tensor(y_pred_last).long().to(args.device)

    accuracy_kmeans = metrics.acc(y_actual, y_pred_last)
    nmi_kmeans = metrics.nmi(y_actual, y_pred_last)
    ari_kmeans = metrics.ari(y_actual, y_pred_last)

    init_centroids = torch.tensor(kmeans.cluster_centers_).float()

    # equal minibatches based on kmeans on z
    x_list = []
    for yi in np.unique(y_pred_last):
        x_list.append(data_tensor[kmeans_y == yi])

    equal_loader = EqualLoader(x_list)

    # Initialize clustering model
    model = Clustering(pretrained_autoencoder=ae_model,
                       n_clusters=args.n_clusters,
                       init_centriods=init_centroids,
                       init_covs = init_covs,
                       args=args,
                       ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate)
    train_loader = 0
    y_pred = 0


    # Loss functions
    criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()


    """
    Lists and Variables
    """
    ae_loss_list = []  # reconstruction loss
    softmax_loss_list = [] # loss from softmax head
    real_epoch = 0
    acc_list = []
    nmi_list = []
    ari_list = []

    epochs_before_update = args.update_interval
    updates_done = 0
    centroids_t0 = model.centroids.clone().detach()
    delta_centroids = 0

    d_mu_steps = []
    w_steps = []
    det_sigma_cov_steps = []

    det_cov_list = []
    full_z_list = []
    y_pred_list = []
    update_epoch_list = []


    # Intialize cluster priors 1 / K times the number of clusters to get uniform cluster weights
    pbar = tqdm(range(args.finetune_epochs+1))
    w = torch.tensor([1/args.n_clusters]*args.n_clusters).float().to(args.device)
    w = w.reshape([1, -1])

    counts = np.bincount(y_pred_last, minlength=args.n_clusters).astype(np.float32)
    w_init = counts / counts.sum()                                   # shape (K,)
    w_kmeans = torch.from_numpy(w_init).float().to(args.device).unsqueeze(0)  # shape (1,K)

    freeze_z = False # Stop factor termination flag to avoid cluster  merging
    got_error = False
    min_w = (1/args.n_clusters) * args.stop_w_factor # default factor = 0.1


    for epoch in pbar:
        gamma = args.gamma

        model.eval()
        if epoch == 0 or epoch==args.finetune_epochs or epochs_before_update == 0:
            epochs_before_update = args.update_interval * 1

            with torch.no_grad():
                full_p, _, full_q = model(data_tensor)
                z, _ = model.ae(data_tensor)

            updates_done += 1

            y_pred = full_p.argmax(1).cpu().numpy()

            w_new = (full_p.sum(dim=0) / (full_p.shape[0])).reshape(1, -1) #/ 100 args.n_clusters*
            w = w if epoch == 0 else w_new
            print(w) # Diagonostics
            freeze_z = w.min() <= min_w

            if epoch==0 or epoch==args.finetune_epochs:
                if (args.update_interval == 1 and epoch % args.plot_interval == 0):
                    update_epoch_list.append(epoch)
                    full_z_list.append(z.cpu().numpy())
                    y_pred_list.append(y_pred)
                else:
                    update_epoch_list.append(epoch)
                    full_z_list.append(z.cpu().numpy())
                    y_pred_list.append(y_pred)

            # Clustering accuracy, NMI, and ARI score computation
            accuracy = metrics.acc(y_actual, y_pred)
            nmi = metrics.nmi(y_actual, y_pred)
            ari = metrics.ari(y_actual, y_pred)
            acc_list.append(accuracy)
            nmi_list.append(nmi)
            ari_list.append(ari)

            # Average mismatch between pred and K-means pred
            # Mismatch between current and previous predictions
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]

            #Update K-means initialized y_pred_last with new y_pred
            y_pred_last = y_pred

            """
            Measuring centroid updates, w is updating, the determinant of sigma is updating during training
            """
            centroids_t1 = model.centroids.clone().detach()
            delta_centroids = (centroids_t1-centroids_t0).pow(2).sum(dim=1).cpu().numpy()
            centroids_t0 = centroids_t1

            ########################## calculate the sigma
            sum_det_sigma_cov_steps = []
            det_sigma_cov = torch.linalg.det(model.get_cov().clone().detach()).cpu().numpy()  # | Σ | (K,)
            det_sigma_cov_steps.append(det_sigma_cov)
            sum_det_sigma_cov = det_sigma_cov.sum()  # sum of the determinants of sigma covariances
            sum_det_sigma_cov_steps.append(sum_det_sigma_cov)

            d_mu_steps.append(delta_centroids)
            w_steps.append(w.cpu().numpy()[0])

            real_epoch = epoch

           # The model stops training after running for finetune_epochs number of epochs
            if epoch == args.finetune_epochs:
                break


            # Stop training when the minor cluster reaches its threshold stop factor
            '''
            When Freeze_z = true, minority cluster is merging dangeorusly - should stop
            '''
            if freeze_z:
                update_epoch_list.append(epoch)
                full_z_list.append(z.cpu().numpy())
                y_pred_list.append(y_pred)
                break

        model.train()


        for batch_x in equal_loader:

            p, x_hat, q = model(batch_x)

            # scale with cluster ratios
            p = p.mul(w)


            ae_loss = criterion(x_hat, batch_x)


            ce_loss = torch.tensor(0.0)
            ce_p = p
            ce_loss = ce_criterion(q, ce_p)

            if not got_error and torch.isnan(ce_loss):
                args.name += 'xxx'
                got_error = True

            # calculate combined loss using reconstruction loss and KL/CE divergence loss
            loss = ae_loss + (gamma * ce_loss)

            ae_loss_list.append(ae_loss.item())
            softmax_loss_list.append(ce_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            with torch.no_grad():
                z, _ = model.ae(data_tensor)

            z_scaler = preprocessing.StandardScaler()
            z = z_scaler.fit_transform(z.cpu().numpy())


            det_cov = []

            for yi in range(args.n_clusters):
                idx = y_pred == yi
                xk = z[idx]
                detk = xk.shape[0] / data_tensor.shape[0]

                det_cov.append(detk)


            det_cov_list.append(det_cov)

            cov_str = ', '.join([f'{det_i:.2f}' for det_i in det_cov])

            pbar.set_description((
                f'Training {args.name} | '
                f'ae_loss: {ae_loss.item():.3f}; '
                f'softmax_loss: {ce_loss.item():.3f}; '
                f'acc: {accuracy:.3f}; ari: {ari:.3f}; '
                f'det_cov: {cov_str}; '
            ))


        if got_error:
            print("stopping training")
            break

        epochs_before_update -= 1

    if not args.timed:
        generate_curves(ae_loss_list, softmax_loss_list, acc_list, det_cov_list,args,
                        d_mu_steps = d_mu_steps,
                        det_sigma_cov_steps = det_sigma_cov_steps,
                        w_steps = w_steps)
        plot_tsnes(data_numpy, full_z_list, y_pred_list, y_actual, args, update_epoch_list)


    metrics_dict = {
        'ae_loss': ae_loss_list,
        'softmax_loss': softmax_loss_list,
        "real_epoch": real_epoch,
        'acc_list': acc_list,
        'accuracy_kmeans': accuracy_kmeans,
        'nmi_kmeans': nmi_kmeans,
        'ari_kmeans': ari_kmeans,
        'det_sigma_cov_steps': det_sigma_cov_steps,
        "w_steps": w_steps,
        "centroids_t1": centroids_t1,
        "w": w,
        'w_kmeans': w_kmeans,
        'full_z_list': full_z_list,
        'y_pred_list': y_pred_list,
        'y_actual': y_actual,
        'update_epoch_list': update_epoch_list,
    }
    return model, metrics_dict


def align_labels(ref, labels):
    """Flip binary labels if flipping gives better agreement with ref."""
    if len(np.unique(labels)) > 2:
        return labels  # skip for multi-class
    if np.sum(labels == ref) >= np.sum((1 - labels) == ref):
        return labels
    return 1 - labels


def mean_to_nearest_label(x, valid_labels):
    """Map an averaged (soft) label back to the nearest valid class label."""
    valid_labels = np.asarray(valid_labels)
    return valid_labels[int(np.argmin(np.abs(valid_labels - x)))]


def ensemble_latent_dims(ld_list, base_name, args, y_actual):
    """Read each latent dimension's saved predictions, align every ld to a
    reference ld (min ld), take the ensemble across all latent
    dimensions, and save the ensemble results."""
    pred_dir = f'./{args.save_file_name}/predictions'
    valid_labels = np.unique(np.asarray(y_actual))

    y_pred = {}
    for ld in ld_list:
        with open(f'{pred_dir}/{base_name}_ld{ld}.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        y_pred[ld] = np.array(data_dict['y_pred'])

    ref_ld_key = min(ld_list)
    ref_ld = np.array(y_pred[ref_ld_key])

    y_pred_aligned = {ld: align_labels(ref_ld, np.array(preds))
                      for ld, preds in y_pred.items()}
    y_pred_pd_flip = pd.DataFrame(y_pred_aligned)
    y_pred_pd_flip['mean'] = y_pred_pd_flip.mean(axis=1)
    y_pred_pd_flip['mean_class_ld'] = y_pred_pd_flip['mean'].apply(
        lambda x: mean_to_nearest_label(x, valid_labels)
    )

    ensemble_pred = y_pred_pd_flip['mean_class_ld'].to_numpy()

    os.makedirs(pred_dir, exist_ok=True)
    y_pred_pd_flip.to_csv(f'{pred_dir}/{base_name}_ensemble.csv', index=False)
    with open(f'{pred_dir}/{base_name}_ensemble.pkl', 'wb') as f:
        pickle.dump({
            'y_pred_per_ld': y_pred,
            'y_pred_aligned': y_pred_aligned,
            'ref_ld': ref_ld_key,
            'ld_list': ld_list,
            'ensemble_pred': ensemble_pred,
            'y_actual': np.asarray(y_actual),
        }, f)

    ens_acc = metrics.acc(y_actual, ensemble_pred)
    ens_nmi = metrics.nmi(y_actual, ensemble_pred)
    ens_ari = metrics.ari(y_actual, ensemble_pred)

    ens_rows = [{
        "dataset": args.data_path if args.data_path else args.dataset,
        "run_name": f'{base_name}_ensemble',
        "ref_ld": ref_ld_key,
        "ld_list": ld_list,
        "acc": ens_acc,
        "nmi": ens_nmi,
        "ari": ens_ari,
    }]
    os.makedirs(f"./{args.save_file_name}/scores", exist_ok=True)
    if args.data_path:
        file_base = os.path.splitext(args.data_path)[0]
        ens_file = f"{args.save_file_name}/scores/ensemble_{file_base}.csv"
    else:
        ens_file = f"{args.save_file_name}/scores/ensemble_{args.dataset}.csv"
    ens_summary = pd.DataFrame(ens_rows)
    if os.path.isfile(ens_file):
        ens_summary.to_csv(ens_file, mode="a", index=False, header=False)
    else:
        ens_summary.to_csv(ens_file, mode="w", index=False, header=True)

    print('Ensemble ld_list:', ld_list)
    print('Ensemble reference ld:', ref_ld_key)
    print('Ensemble ACC:', ens_acc)
    print('Ensemble NMI:', ens_nmi)
    print('Ensemble ARI:', ens_ari)

    return ensemble_pred


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    from utils.openml import get_data

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='1510',
                        help="give openml dataset id (default: breast-cancer)")

    parser.add_argument('--data_path', type=str, default=None,
                        help="Path to local dataset (CSV/Parquet etc.)")
    parser.add_argument('--save_file_name', type=str, default=None,
                        help="Folder name under which results are saved")

    parser.add_argument('--n_clusters', default=0, type=int,
                        help='Number of clusters, 0 = no of classes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--latent_dim', default=10, type=int,
                        help='dimension of bottleneck layer')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int,
                        help='updates target (P) at set intervals')
    parser.add_argument('--finetune_epochs', default=1e4, type=int)
    parser.add_argument('--l_rate', default=0.00001, type=float,
                        help="Learning Rate default 1e-5")
    parser.add_argument('--pretrain_epochs', default=1e4, type=int)
    parser.add_argument('--device', default='cpu', type=str,
                        help="use 'cuda:0' to select cuda devices")
    parser.add_argument('--name', type=str, default='model',
                        help="model nickname)")
    parser.add_argument('--debug_autograd', action='store_true',
                        help='debug autograd')
    parser.add_argument('--print', action='store_true',
                        help='print-friendly plots')
    parser.add_argument('--timed', action='store_true',
                        help='skip plots & prints to save compute cycles')
    parser.add_argument('--plot_all_tsne', action='store_true',
                        help='plot tnse for all updates')
    parser.add_argument('--redo_pretrain', action='store_true',
                        help='redo pretraining of AE')
    parser.add_argument('--reducer', default='tsne', type=str,
                    choices=['tsne', 'umap', 'pca'])
    parser.add_argument('--plot_interval', default=100, type=int,
                        help='')
    parser.add_argument('--tsne_dim', default=2, type=int,
                        help='')
    parser.add_argument('--stop_w_factor', default=0.1, type=float)
    args = parser.parse_args()
    args.dataset = int(args.dataset)
    args.pretrain_epochs = int(args.pretrain_epochs)
    args.finetune_epochs = int(args.finetune_epochs)

    torch.autograd.set_detect_anomaly(args.debug_autograd)
    if args.debug_autograd:
        torch.set_printoptions(threshold=10_000)

    if args.data_path:
        df = pd.read_csv(args.data_path)
        df = df.drop(columns=['index', 'Unnamed: 0', 'level_0'], errors='ignore')
        y_actual = df['class']
        X_actual = df.drop(columns=['class'])
        X_actual = X_actual.fillna(X_actual.median())
    else:
        X_actual, y_actual = get_data(args.dataset)

    n_classes = len(np.unique(y_actual))

    args.n_clusters = n_classes if args.n_clusters == 0 else args.n_clusters

    # Standardize data
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X_actual)
    data_tensor = torch.from_numpy(X).float().to(args.device)


    args.batch_count = np.ceil(X.shape[0] / args.batch_size)

    print('sample size', X.shape[0], 'batch_size', args.batch_size, 'batch count', args.batch_count)

    # build the latent-dimension candidate list.
    n_features = X.shape[1]
    if n_features < 30:
        ld_list = list(range(2, n_features + 1, 3))
    else:
        ld_list = sorted(set(int(round(v)) for v in np.linspace(2, n_features, 10)))
    print('latent dimension candidates:', ld_list)

    base_name = args.name

    for ld in ld_list:
        args.latent_dim = ld
        args.name = f'{base_name}_ld{ld}'
        start_time = time.time()
        print(f'\n===== latent_dim = {ld} ({args.name}) =====')

        # Pretrain an Autoencoder
        ae_model = pretrain(data_tensor, args)

        # Finetune the autoencoder with joint clustering loss
        model, metrics_dict = train(ae_model, data_tensor, y_actual, args)

        # final acc
        model.eval()
        with torch.no_grad():
            z, _ = model.ae(data_tensor)
            full_p, _, full_q = model(data_tensor)

        z = z.cpu().numpy()

        y_pred = full_p.argmax(1).cpu().numpy()

        end_time = time.time()

        acc = metrics.acc(y_actual, y_pred)
        nmi = metrics.nmi(y_actual, y_pred)
        ari = metrics.ari(y_actual, y_pred)

        h = np.sum(z)
        print(f"latent_checksum={h:.6f}")

        os.makedirs(f"./{args.save_file_name}/predictions", exist_ok=True)
        save_data = {
            'z': z,
            'sum_z': h,
            "y_actual": y_actual,
            "y_pred": y_pred,
            "metrics": metrics_dict,
            "args": vars(args),
        }
        out_name = f'./{args.save_file_name}/predictions/{args.name}.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(save_data, f)

        rows = [{
            "ld": args.latent_dim,
            "dataset": args.data_path if args.data_path else args.dataset,
            "run_name": args.name,
            "gamma": args.gamma,
            "l_rate": args.l_rate,
            "update_interval": args.update_interval,
            "batch_size": args.batch_size,
            'batch_count': args.batch_count,
            "pretrain_epochs": int(args.pretrain_epochs),
            "finetune_epochs": int(args.finetune_epochs),
            "stop_w_factor": args.stop_w_factor,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "time(s)": end_time - start_time,
            'accuracy_kmeans': metrics_dict['accuracy_kmeans'],
            'nmi_kmeans': metrics_dict['nmi_kmeans'],
            'ari_kmeans': metrics_dict['ari_kmeans'],
            "real_epoch": metrics_dict['real_epoch'],
            "acc": acc,
            "nmi": nmi,
            "ari": ari,
            'w_kmeans': metrics_dict['w_kmeans'].tolist(),
            'w': metrics_dict['w'].tolist(),
        }]

        if args.data_path:
            file_base = os.path.splitext(args.data_path)[0]
            file_name = f"{args.save_file_name}/scores/results_dimstop_{file_base}.csv"
        else:
            file_name = f"{args.save_file_name}/scores/results_dimstop_{args.dataset}.csv"
        os.makedirs(f"./{args.save_file_name}/scores", exist_ok=True)
        summary = pd.DataFrame(rows)

        if os.path.isfile(file_name):
            summary.to_csv(file_name, mode="a", index=False, header=False)
        else:
            summary.to_csv(file_name, mode="w", index=False, header=True)

        print('Dataset:', args.dataset)
        print('Latent Dimension:', args.latent_dim)
        print('n_samples:', data_tensor.shape[0])
        print('n_features:', data_tensor.shape[1])
        print('n_classes:', n_classes)
        print('n_clusters:', args.n_clusters)
        print('Time:', end_time - start_time)
        print('ACC:', acc)
        print('NMI:', nmi)
        print('ARI:', ari)
        print('accuracy_kmeans', metrics_dict['accuracy_kmeans'])
        print('nmi_kmeans', metrics_dict['nmi_kmeans'])
        print('ari_kmeans', metrics_dict['ari_kmeans'])

    ensemble_latent_dims(ld_list, base_name, args, y_actual)
