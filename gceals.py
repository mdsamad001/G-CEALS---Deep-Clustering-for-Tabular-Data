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

# subnetwork attached to bottleneck
#


class Clustering(nn.Module):
    def __init__(self, pretrained_autoencoder, n_clusters,
                 init_centriods=None, init_covs=None, alpha=1, args=None):
        super().__init__()
        self.ae = pretrained_autoencoder
        self.n_clusters = n_clusters
        self.alpha = alpha
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
        

        self.diag_mask = torch.eye(self.ae.output_level, device=args.device).unsqueeze(0).expand(init_covs.shape)
        self.identity_var = torch.eye(self.ae.output_level, dtype=torch.float, device=args.device)
        diag = torch.arange(self.ae.output_level)
        init_covs_diag = init_covs[:, diag, diag].reshape(-1, 1, diag.shape[0])
        # softplus(x) = 1/B log_e (1 + exp(B*x))
        softplus_reverse = torch.log(torch.exp(init_covs_diag * 1) - 1)
            
        # Cov = I + softplus(add_var)
        self.add_var = nn.Parameter(softplus_reverse)
        
        activation = nn.ReLU
        
        self.softmax_head = nn.Sequential(
            nn.Linear(self.ae.output_level, self.ae.output_level),
            activation(),
            nn.Linear(self.ae.output_level, self.n_clusters),
        )
        

    def get_cov(self):
        # Cov = I + softplus(add_var)
        sigma = self.identity_var*F.softplus(self.add_var)
        return sigma

    def get_idec_q(self, z):
        '''IDEC cluster assignment using t-distribution'''
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) -
                   self.centroids, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def get_gceals_q(self, z):
        '''compute q using softmax of malahalobis'''
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
        d4 = d3.pow(0.5)
        d_final = d4
        softmax_q = F.softmax(-d_final, dim=1)
        
        return softmax_q
    
    def softmax_output(self, z):
        q2 = self.softmax_head(z)
        return q2
    
    def cluster_output(self, z):
        q1 = self.get_gceals_q(z)
        return q1

    def forward(self, x):
        z, x_hat = self.ae(x)
        
        q1 = self.cluster_output(z)
        q2 = self.softmax_output(z)
            
        return q1, x_hat, q2


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
    kl_loss_list = []  # kldiv loss without gamma scale

    # prepare for pretraining
    # lr = args.l_rate
    lr = 1e-3  # fixed it since we reuse the same pretrained autoencoder model
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ae_loss_list.append(ae_loss.item())
            kl_loss_list.append(0)

            pbar.set_description((
                f'Pretraining | ae_loss: {ae_loss.item():.5f};'
            ))

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
    

    # Finetune clustering_model
    criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    nll_criterion = nn.NLLLoss()
    kl_criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
    ae_loss_list = []  # reconstruction loss
    kl_loss_list = []  # kldiv loss without gamma scale 
    softmax_loss_list = [] # loss from softmax head
    acc_list = []

    x_hat_prev = 0
    q_prev = 0
    epochs_before_update = args.update_interval
    updates_done = 0
    centroids_t0 = model.centroids.clone().detach()
    delta_centroids = 0
    
    d_mu_steps = []
    w_steps = []
    det_sigma_steps = []
    
    det_cov_list = []
    full_z_list = []
    y_pred_list = []
    update_epoch_list = []

    pbar = tqdm(range(args.finetune_epochs+1))
    w = torch.tensor([1/args.n_clusters]*args.n_clusters).float().to(args.device)
    w = w.reshape([1, -1])
    
    freeze_z = False
    got_error = False
    min_w = (1/args.n_clusters) * args.stop_w_factor # default factor = 0.5
    
    
    for epoch in pbar:
        gamma = args.gamma
        
        model.eval()
        if epoch == 0 or epoch==args.finetune_epochs or epochs_before_update == 0:
            epochs_before_update = args.update_interval * 1
            
            with torch.no_grad():
                full_q, _, full_q2 = model(data_tensor)
                z, _ = model.ae(data_tensor)
            
            updates_done += 1

            y_pred = full_q.argmax(1).cpu().numpy()
            
            w_new = (full_q.sum(dim=0) / (full_q.shape[0])).reshape(1, -1) #/ 100 args.n_clusters*
            w = w if epoch == 0 else w_new
            print(w)
            freeze_z = w.min() <= min_w
            
            if epoch==0 or epoch==args.finetune_epochs or args.plot_all_tsne:
                if (args.update_interval == 1 and epoch % args.plot_interval == 0):
                    update_epoch_list.append(epoch)
                    full_z_list.append(z.cpu().numpy())
                    y_pred_list.append(y_pred)
                else:
                    update_epoch_list.append(epoch)
                    full_z_list.append(z.cpu().numpy())
                    y_pred_list.append(y_pred)

            accuracy = metrics.acc(y_actual, y_pred)
            ari = metrics.ari(y_actual, y_pred)
            acc_list.append(accuracy)

            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            
            centroids_t1 = model.centroids.clone().detach()
            delta_centroids = (centroids_t1-centroids_t0).pow(2).sum(dim=1).cpu().numpy()         
            centroids_t0 = centroids_t1
            
            d_mu_steps.append(delta_centroids)
            w_steps.append(w.cpu().numpy()[0])
            det_sigma = torch.linalg.det(model.get_cov().clone().detach()).cpu().numpy()
            det_sigma_steps.append(det_sigma)
                
            # need this manual break to plot last acc point
            if epoch == args.finetune_epochs:
                break
                
            if freeze_z: 
                update_epoch_list.append(epoch)
                full_z_list.append(z.cpu().numpy())
                y_pred_list.append(y_pred)
                break

        model.train()
        
        
        for batch_x in equal_loader:
                
            q, x_hat, q2 = model(batch_x)
                    
            # scale with cluster ratios
            q = q.mul(w)
            

            ae_loss = criterion(x_hat, batch_x)
                            
            
            ce_loss = torch.tensor(0.0)
            ce_p = q 
            ce_loss = ce_criterion(q2, ce_p)

            if not got_error and torch.isnan(ce_loss):
                args.name += 'xxx'
                got_error = True

            # calculate combined loss using reconstruction loss and KL divergence loss
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

            x_hat_prev = x_hat.detach().cpu()
            q_prev = q.detach().cpu()

        if got_error:
            print("stopping training")
            break
            
        epochs_before_update -= 1

    if not args.timed:
        generate_curves(ae_loss_list, kl_loss_list, softmax_loss_list, acc_list, det_cov_list,
                        d_mu_steps = d_mu_steps,
                        det_sigma_steps = det_sigma_steps,
                        w_steps = w_steps,
                        args = args)
        plot_tsnes(data_numpy, full_z_list, y_pred_list, y_actual, args, update_epoch_list)
    

    return model


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    from utils.openml import get_data

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='1510',
                        help="give openml dataset id (default: breast-cancer)")
    parser.add_argument('--n_clusters', default=0, type=int,
                        help='Number of clusters, 0 = no of classes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--latent_dim', default=10, type=int,
                        help='dimension of bottleneck layer')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int,
                        help='updates target (P) at set intervals')
    parser.add_argument('--finetune_epochs', default=1e3, type=int)
    parser.add_argument('--l_rate', default=0.001, type=float,
                        help="Learning Rate")
    parser.add_argument('--pretrain_epochs', default=1e3, type=int)
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
    parser.add_argument('--stop_w_factor', default=0.5, type=float)
    args = parser.parse_args()
    args.dataset = int(args.dataset)
    args.pretrain_epochs = int(args.pretrain_epochs)
    args.finetune_epochs = int(args.finetune_epochs)

    torch.autograd.set_detect_anomaly(args.debug_autograd)
    if args.debug_autograd:
        torch.set_printoptions(threshold=10_000)

    start_time = time.time()

    X_actual, y_actual = get_data(args.dataset)
    n_classes = len(np.unique(y_actual))
    
    args.n_clusters = n_classes if args.n_clusters == 0 else args.n_clusters

    # Standardize data
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X_actual)
    data_tensor = torch.from_numpy(X).float().to(args.device)
    
    
    args.batch_count = np.ceil(X.shape[0] / args.batch_size)
    
    print(X.shape[0], args.batch_size, X.shape[0]/args.batch_size, args.batch_count)

    ae_model = pretrain(data_tensor, args)

    model = train(ae_model, data_tensor, y_actual, args)

    # final acc
    model.eval()
    with torch.no_grad():
        z, _ = model.ae(data_tensor)
        full_q, _, full_q2 = model(data_tensor)
        
    z = z.cpu().numpy()

    y_pred = full_q.argmax(1).cpu().numpy()

    from utils.pickle import save_var
    save_var(y_pred, f'./predictions/{args.name}.pkl')

    end_time = time.time()

    acc = metrics.acc(y_actual, y_pred)
    nmi = metrics.nmi(y_actual, y_pred)
    ari = metrics.ari(y_actual, y_pred)

    print('Dataset:', args.dataset)
    print('n_samples:', data_tensor.shape[0])
    print('n_features:', data_tensor.shape[1])
    print('n_classes:', n_classes)
    print('n_clusters:', args.n_clusters)
    print('Time:', end_time - start_time)
    print('ACC:', acc)
    print('NMI:', nmi)
    print('ARI:', ari)
