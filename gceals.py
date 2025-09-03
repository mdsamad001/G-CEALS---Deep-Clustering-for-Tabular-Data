from functools import partial
import time
from utils import metrics, tabular_data
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
from utils.corruptor import Corruptor
from utils.dataloader import EqualLoader
from utils.pickle import save_var


torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    print("CUDA available")


class JSD(nn.Module):
    '''
    Jensen–Shannon divergence (JSD) between two probabilities P & Q.
    there is no official pytorch implementation
    thus we source from the pytorch community: 
    https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/11
    '''

    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        # note: pytorch convention is kl(pred, true)
        # thus p = pred and q = true
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

# IDEC target
def idec_p(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def depict_p(q):
    weight = q / q.sum(0).pow(0.5)
    return (weight.t() / weight.sum(1)).t()

# GCEAL target P (and also Q without numerical error fix)
def gceals_cluster_assignment(z, centroids, sigma_inv=None):
    '''compute q using softmax of malahalobis'''
    x = z.unsqueeze(dim=1)  # (n, d) -> (n, 1, d)
    mu = centroids.unsqueeze(dim=0)  # (k, d) -> (1, k, d)

    d = x - mu  # (n, 1, d) - (1, k, d) -> (n, k, d)
    d1 = d.unsqueeze(dim=2)  # (n, k, d) -> (n, k, 1, d)
    # (n, k, 1, d) \times (n, k, d, d) -> (n, k, 1, d)
    d2 = d1.matmul(sigma_inv) if torch.is_tensor(sigma_inv) else d1
    # (n, k, 1, d) \times (n, k, d, 1) -> (n, k, 1, 1) -> (n, k, 1)
    d3 = d2.matmul(d1.transpose(2, 3)).squeeze()
    # S = softmax of mahalanobis distances
    d4 = d3.pow(0.5)
    # test
    # d4_min, _ = torch.min(d4, dim=0)
    # d_final = d4 - d4_min 
    # ----
    d_final = d4 # actual
    # ----
    q = F.softmax(-d_final, dim=1)
    return q


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
        self.alpha = alpha  # used by idec/dec's t-dist
        self.use_svd = args.use_svd  # use low-rank approx of sigma
        self.use_gceals_q = args.gceals_q
        self.use_gceals_p = args.gceals_p
        self.use_sigma = args.sigma
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
        
        self.static_centroids = init_centriods.to(args.device)

        self.covs = nn.Parameter(init_covs)
        self.diag_mask = torch.eye(self.ae.output_level, device=args.device).unsqueeze(
            0).expand(self.covs.shape)
        
        self.identity_var = torch.eye(self.ae.output_level, dtype=torch.float, device=args.device)
        # v2 uses torch.zeros; v1 ones
        diag = torch.arange(self.ae.output_level)
        # add_var_init = init_covs[:, diag, diag].reshape(-1, 1, diag.shape[0]) - torch.ones(n_clusters, 1, self.ae.output_level)
        # add_var_init = add_var_init.pow(0.5)
        # add_var_init = torch.ones(n_clusters, 1, self.ae.output_level)
        
        init_covs_diag = init_covs[:, diag, diag].reshape(-1, 1, diag.shape[0])
        # softplus(x) = 1/B log_e (1 + exp(B*x))
        softplus_reverse = torch.log(torch.exp(init_covs_diag * 1) - 1)
        
        
        # if args.test_covs:
        #     self.add_var = nn.Parameter(add_var_init) 
        if args.reverse_softplus and args.test_covs2:
            add_var_init = softplus_reverse
        else: 
            add_var_init = torch.ones(n_clusters, 1, self.ae.output_level)
            
        self.add_var = nn.Parameter(softplus_reverse)
        
        
        # print(self.add_var.shape)#, self.add_var*self.identity_var).shape)
        # exit()
        
        # print('got init cov\n', init_covs)
        # print('decommposed to I + v, v:\n', add_var_init)
        # exit()

        # use svd for cov
        U, S, Vt = torch.linalg.svd(init_covs)
        self.U = nn.Parameter(U.float())

        # prevent linAlg error by
        # remove hard probs ([1, 0]) from q
        # problem still exist on model = gceals-q+sigma on dataset=1063
        self.temp = args.temp
        self.eps = args.eps
        self.eps_max = 1-(self.eps*(self.n_clusters-1))
        
        activation = nn.ReLU
        # activation = partial(nn.LeakyReLU, 0.2)
        
        self.softmax_head = nn.Sequential(
            nn.Linear(self.ae.output_level, self.ae.output_level),
            activation(),
            nn.Linear(self.ae.output_level, self.n_clusters),
        )
        
        self.w = nn.Parameter(torch.ones(1, n_clusters)*1/args.n_clusters)
        
    def freeze_centroids(self):
        self.centroids.requires_grad = False
        
    def unfreeze_centroids(self):
        self.centroids.requires_grad = True
        
    def update_static_centroids(self, z, q):
        y = q.argmax(dim=1)
        
        x = z.unsqueeze(dim=1)  # (n, d) -> (n, 1, d)
        centroids = self.static_centroids
        mu = centroids.unsqueeze(dim=0)  # (k, d) -> (1, k, d)
        d = x - mu  # (n, 1, d) - (1, k, d) -> (n, k, d)
        # print(d.shape, d[:,0,:].shape)
        
        for k in range(self.args.n_clusters):
            m = y == k
            # print(m.sum(), d[m, k, :].shape)
            # total distance from centroid to all cluster members
            delta_centroid = d[m, k, :].sum(dim=0).div(1+m.sum())
            centroids[k] += delta_centroid
            
        self.static_centroids = centroids
        

    def get_cov(self):
        if not self.use_sigma:
            # TypeError: pinverse(): argument 'input' (position 1) must be Tensor, not int
            return torch.tensor(self.args.scalar_sigma).float()
        # compute sigma from U@Ut
        if self.use_svd:
            sigma = self.U@self.U.transpose(1, 2) 
        elif self.args.test_covs:
            sigma = self.identity_var + self.identity_var*self.add_var.pow(2) # num-errors 2/16
            # sigma = self.identity_var*self.add_var # num-errors  
        elif self.args.test_covs2: # v2 works but not good as no-sigma
            sigma = self.identity_var*F.softplus(self.add_var) 
        else:
            sigma = self.covs
        # # sigma = sigma.mul(self.diag_mask) # use diagonals only
        # # print(sigma)
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
        centroids = self.static_centroids if self.args.static_centroids else self.centroids
        mu = centroids.unsqueeze(dim=0)  # (k, d) -> (1, k, d)

        sigma_inv = torch.pinverse(self.get_cov()) if self.use_sigma else torch.tensor(1).float()

        d = x - mu  # (n, 1, d) - (1, k, d) -> (n, k, d)
        d1 = d.unsqueeze(dim=2)  # (n, k, d) -> (n, k, 1, d)
        # (n, k, 1, d) \times (n, k, d, d) -> (n, k, 1, d)
        d2 = d1.matmul(sigma_inv) if self.use_sigma else d1
        # (n, k, 1, d) \times (n, k, d, 1) -> (n, k, 1, 1) -> (n, k, 1)
        d3 = d2.matmul(d1.transpose(2, 3)).squeeze()
        # S = softmax of mahalanobis distances
        d4 = d3.pow(0.5)
        
        # test
        # d4_min, _ = torch.min(d4, dim=0) 
        # d_final = d4 - d4_min 
        # ----
        d_final = d4 # actual

        softmax_q = F.softmax(-d_final/self.temp, dim=1)
        # softmax_q = torch.clamp(softmax_q, min=self.eps, max=self.eps_max) # remove hard probs
        
        if args.debug_autograd and torch.any(torch.isnan(q)):
            raise SyntaxError

        return softmax_q
    
    def softmax_output(self, z):
        q2 = self.softmax_head(z)
        if self.args.nll:
            q2 = F.softmax(q2, dim=1)
            # q2 = torch.clamp(q2, min=self.eps, max=self.eps_max) 
            
        return q2#*self.w
    
    def cluster_output(self, z):
        q1 = self.get_gceals_q(z) if self.use_gceals_q else self.get_idec_q(z)
        # q1 = self.get_idec_q(z)
        
        return q1

    def forward(self, x):
        z, x_hat = self.ae(x)
        
        q1 = self.cluster_output(z) #posterior probability distribution over all clusters 
        q2 = self.softmax_output(z) #softmax predictions
            
        return q1, x_hat, q2

def cont_loss(aug_features_1, aug_features_2, mlp1, mlp2, nce_temp=1):
    aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True))#.flatten(1,2)
    aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True))#.flatten(1,2)
    
    aug_features_1 = mlp1(aug_features_1)
    aug_features_2 = mlp2(aug_features_2)
    
    logits_per_aug1 = aug_features_1 @ aug_features_2.t()/nce_temp
    logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/nce_temp
    targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
    loss_1 = F.cross_entropy(logits_per_aug1, targets)
    loss_2 = F.cross_entropy(logits_per_aug2, targets)
    loss   = (loss_1 + loss_2)/2
    
    return loss

def gini_loss(probs):
    g = probs
    loss = g.mul(1-g).sum(dim=0).sum()
    
    return loss

def centroid_loss(centroids):
    u = (centroids / centroids.norm(dim=-1, keepdim=True))#.flatten(1,2)
    uu = u.matmul(u.T)
    # print('uu shape', uu.shape, uu)
    # exit()
    I = torch.eye(u.shape[0]).to(centroids.device)
    
    uu_abs = torch.abs(uu)
    
    # loss = (uu_abs - I).sum(dim=0).sum()
    loss = F.cross_entropy(uu, I)
    
    return loss

def pretrain(data_tensor, args):
    # Initialize model
    ae_model = Autoencoder(input_level=data_tensor.size()[1],
                           output_level=args.latent_dim).to(args.device)
    
    corruptor_str = '' if args.corruptor_type == 'pass' else f'_{args.corruptor_type}-{args.corruptor_rate}'
    file_name = f'pretrained-ae_dim-{args.latent_dim}_epoch-{args.pretrain_epochs}_{args.dataset}{corruptor_str}'
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
    
    corruptor = Corruptor(data_tensor, args.corruptor_rate, args.corruptor_type)
    

    # Pretrain ae_model
    ae_model.train()
    pbar = tqdm(range(args.pretrain_epochs))
    for epoch in pbar:
        for batch_x in pretrain_loader:
            noisy_batch_x = corruptor(batch_x)
            
            # batch_z, x_hat = ae_model(batch_x)
            batch_z, x_hat = ae_model(noisy_batch_x)            
            
            
            # y1 = ae_model.classifier1(batch_x)
            
            # print(y1)
            # print(y2)
            
            # exit()
            
            bce_loss = torch.tensor(0.0)
            # bce_loss = cont_loss(batch_x,batch_z, ae_model.classifier1, ae_model.classifier2)
            
            ae_loss = criterion(x_hat, batch_x)
            # bce_loss = F.binary_cross_entropy(y2, y1)
            
            
            loss = ae_loss + bce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ae_loss_list.append(ae_loss.item())
            kl_loss_list.append(bce_loss.item())

            pbar.set_description((
                f'Pretraining | ae_loss: {ae_loss.item():.5f}; aux_loss: {bce_loss.item():.5f};'
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
    
    save_var(ae_loss_list, f'./debug/{file_name}_loss.pkl')

    return ae_model


def train(ae_model, data_tensor, y_actual, args):

    
    
    method = 'gceals' if args.gceals_p or args.gceals_q else 'idec'
    # method += '_gmm' if args.gmm_mu or args.gmm_sigma else ''
    # method += '-mu' if args.gmm_mu else ''
    # method += '-sigma' if args.gmm_sigma else ''
    # method += '-svd' if args.use_svd else ''
    file_name = f'{method}_dim-{args.latent_dim}_epoch-{args.finetune_epochs}_gamma={args.gamma}_{args.dataset}'

    # compute cluster centroid, sigma on X for gceals P
    kmeans_on_x = KMeans(n_clusters=args.n_clusters,
                         init="k-means++", n_init='auto', random_state=42)
    data_numpy = data_tensor.cpu().numpy()
    kmeans_on_x.fit(data_numpy)
    centroids_on_x = torch.tensor(
        kmeans_on_x.cluster_centers_, requires_grad=False).float().to(args.device)
    covs_inv_on_x = None

    # gceals p with GMM
    if args.gceals_p_gmm:
        gmm_on_x = GaussianMixture(n_components=args.n_clusters,
                                   init_params="k-means++",
                                   random_state=42)
        gmm_on_x.fit(data_numpy)
        centroids_on_x = torch.tensor(
            gmm_on_x.means_, requires_grad=False).float().to(args.device)
        covs_on_x = torch.tensor(
            gmm_on_x.covariances_, requires_grad=False).float().to(args.device)
        covs_inv_on_x = torch.pinverse(covs_on_x)

    # we do not need gradients here as it has no parameters
    gceals_p = gceals_cluster_assignment(
        data_tensor, centroids_on_x, covs_inv_on_x).detach()

    # get Z from pretrained AE
    ae_model.eval()
    with torch.no_grad():
        Z_tensor, _ = ae_model(data_tensor)
        Z = Z_tensor.cpu().numpy()

    # initialize centroids with Z
    init_centroids = None
    init_covs = None
    
    
    if args.init_on_tsne and data_tensor.shape[1] > args.latent_dim * 1.0: 
    
        # test kmeans on tsne embeddings and use cluster labels to find centroids on z-space
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=args.tsne_dim, learning_rate='auto', init='random')
        Z_tsne = tsne.fit_transform(data_numpy)

        kmeans = KMeans(n_clusters=args.n_clusters,
                        init="k-means++", n_init='auto', random_state=42)
        y_pred_last = kmeans.fit_predict(Z_tsne)
        kmeans_y = torch.tensor(y_pred_last).long().to(args.device)
        manual_centroids = []

        for i in range(args.n_clusters):
            m = y_pred_last == i

            manual_centroids.append(Z_tensor[m].mean(dim=0).reshape(1, -1))
    
        # print(manual_centroids, torch.concat(manual_centroids, dim=0).float())
        init_centroids = torch.concat(manual_centroids, dim=0).float()
    ## end test
    else:

        kmeans = KMeans(n_clusters=args.n_clusters,
                        init="k-means++", n_init='auto', random_state=42) #args.seed
        y_pred_last = kmeans.fit_predict(Z)
        kmeans_y = torch.tensor(y_pred_last).long().to(args.device)

        init_centroids = torch.tensor(kmeans.cluster_centers_).float()
    

    # initialize covariance with gmm (TODO: IF might save some compute)
    gmm = GaussianMixture(n_components=args.n_clusters,
                          init_params="k-means++",
                          random_state=42)
    _ = gmm.fit_predict(Z)
    init_centroids = torch.tensor(
        gmm.means_).float() if args.gmm_mu else init_centroids
    init_covs = torch.tensor(gmm.covariances_).float(
    ) if args.gmm_sigma else init_covs
    
    # print(init_covs)
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
    
    # with torch.no_grad():
    #     gceals_p, _ = model(data_tensor)

    # Finetune clustering_model
    criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    nll_criterion = nn.NLLLoss()
    kl_criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
    js_criterion = JSD()
    ae_loss_list = []  # reconstruction loss
    kl_loss_list = []  # kldiv loss without gamma scale 
    softmax_loss_list = [] # loss from softmax head
    acc_list = []

    got_error = False

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
    w = w.reshape([1, -1])#.repeat(data_tensor.shape[0], 1)
    
    freeze_z = False
    
    for epoch in pbar:
        gamma = args.gamma
        alpha = args.alpha
        g_lambda = 0.01
        g_lambda2 = 1.0
        # gamma = 0 epoch < arg.delay_gamma else args.gamma
        # epoch < args.delay_gamma or 
        
        # gamma = args.gamma if freeze_z else 0
        # alpha = 0 if freeze_z else args.alpha
        
        model.eval()
        # print(model.get_cov())
        if epoch == 0 or epoch==args.finetune_epochs or epochs_before_update == 0:
            epochs_before_update = args.update_interval * 1
            with torch.no_grad():
                full_q, _, full_q2 = model(data_tensor)
                z, _ = model.ae(data_tensor)
            
            full_p = gceals_p if (args.gceals_p) else depict_p(full_q) if args.depict_p else idec_p(full_q)
            updates_done += 1
            
            if args.static_centroids:
                model.update_static_centroids(z, full_q)
            
            ##  test; only intialize once so that the batching seed stay same
            if epoch == 0 or args.update_full_p:
                
                g = torch.Generator()
                g.manual_seed(epoch)
                # print(data_tensor.shape, full_p.shape, kmeans_y.shape)
                train_loader = DataLoader(TensorDataset(
                    data_tensor, full_p, kmeans_y), batch_size=args.batch_size, shuffle=True, generator=g)

            y_pred = full_q.argmax(1).cpu().numpy()
            # for yi in range(args.n_clusters):
                # idx = y_pred == yi
                # w[0, yi] = w[0,yi] if epoch==0 else idx.sum() / y_pred.shape[0]
            
            full_q = full_q.mul(w) if args.update_using_wq else full_q
            w_new = (full_q.sum(dim=0) / (full_q.shape[0])).reshape(1, -1) #/ 100 args.n_clusters*
            w = w if epoch == 0 else w_new
            
            # freeze if minor cluster is 1/4th of w_o; w_o = n / k
            # min_w = 0.20*(full_q.shape[0] / args.n_clusters)
            min_w = 1/args.n_clusters * args.stop_w_factor # default factor = 0.5
            freeze_z = w.min() <= min_w
            # print(w, w.sum(dim=1))
            # exit()
            
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
            nmi = metrics.nmi(y_actual, y_pred)
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
            # print(det_sigma)
            # exit()
            det_sigma_steps.append(det_sigma)

            # cov_str = ', '.join([f'{det_i:.5f}' for det_i in det_cov])
            if args.early_stop and epoch > 0 and delta_label < args.tol:
                pbar.set_description((
                    f'Training | '
                    f'ae_loss: {ae_loss.item():.5f}; kl_loss: {kl_loss.item():.5f}; '
                    f'acc: {accuracy:.5f}; ari: {ari:.5f}; '
                    # f'delta_label: {delta_label:.5f}; delta_centroids: {delta_centroids:.5f}; Early Stop'
                    # f'det_cov: {cov_str}; '
                ))
                break
                
            # need this manual break to plot last acc point
            if epoch == args.finetune_epochs:
                break
                
            if args.early_stop_w and freeze_z: 
                update_epoch_list.append(epoch)
                full_z_list.append(z.cpu().numpy())
                y_pred_list.append(y_pred)
                break

        model.train()
        # if epoch < 20: continue
        
        # for batch_x, batch_p, batch_y in train_loader:
        for batch_x in equal_loader: ## expt v2
            # if freeze_z:
            #     z, x_hat = model.ae(batch_x)
            #     # x_hat = model.ae.decoder(z.detach())
            #     # train heads only first
            #     q = model.cluster_output(z.detach())
            #     q2 = model.softmax_output(z.detach())
            # else:
            #     q, x_hat, q2 = model(batch_x)
                
            q, x_hat, q2 = model(batch_x)
            
            if args.w_minibatch:
                    w = q.sum(dim=0)/q.shape[0]
                    
            if args.use_w:
                q = q.mul(w)
                q = q / q.sum(dim=1).reshape(-1, 1).repeat(1,q.shape[1]) if args.norm_q else q
                
            if args.use_w_softmax:
                q2 = q2.mul(w)
                

            ae_loss = criterion(
                x_hat, batch_x) if not args.dec else torch.tensor(0)
                            
            # TODO refractor rename var: kl_loss
            kl_loss = torch.tensor(0.0)
            batch_p = depict_p(q) if args.depict_p else idec_p(q)
            kl_loss = kl_criterion(
                q.log(), batch_p) if not args.jsd else js_criterion(q, batch_p)
            
            
            
            
            kl_loss2 = torch.tensor(0.0)
            if args.ce or args.nll:
                ce_p = q if epoch>args.delay_alpha else q.detach() # v0
                if args.ce_p:
                    ce_p = depict_p(q) if args.depict_p else idec_p(q) # v1
                
                kl_loss2 = ce_criterion(q2, ce_p) if not args.nll else nll_criterion(q2.log(), ce_p.argmax(dim=1))

            if not got_error and torch.isnan(kl_loss2):
                args.name += 'xxx'
                got_error = True

            # calculate combined loss using reconstruction loss and KL divergence loss
            if not args.dec:
                # cov_loss = (1-torch.linalg.det(model.get_cov()).min()).pow(2)
                # cov_loss +=  (3-torch.linalg.det(model.get_cov()).sum()).pow(2)
                # cov_loss = F.mse_loss(w, model.w)
                cov_loss = torch.tensor(0.0).to(args.device)
                c1_loss = torch.tensor(0.0).to(args.device)
                c2_loss = torch.tensor(0.0).to(args.device)
                
                if args.centroid_loss:
                    c1_loss = centroid_loss(model.centroids)
                    # c1_loss = cont_loss(model.centroids, model.centroids)
                    cov_loss += c1_loss
                if args.gini_loss:
                    c2_loss = gini_loss(q)
                    cov_loss += c2_loss
                loss = ae_loss + (gamma * kl_loss) + (alpha * kl_loss2) + cov_loss
                # loss = ae_loss + (alpha * kl_loss2) # dont use kld from cluster head / test
            else:
                loss = (kl_loss)
                
                
            if args.plot_comp_graph:
                from graphviz import Digraph
                from torch.autograd import Variable
                # make_dot was moved to https://github.com/szagoruyko/pytorchviz
                from torchviz import make_dot
                
                make_dot(loss, params=dict(model.named_parameters())).render(f'./debug/{args.name}_cg', format="png")
                exit()

            ae_loss_list.append(ae_loss.item())
            kl_loss_list.append(kl_loss.item())
            softmax_loss_list.append(kl_loss2.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_name = 'js_loss' if args.jsd else 'kl_loss'
            
            with torch.no_grad():
                z, _ = model.ae(data_tensor)
                
            z_scaler = preprocessing.StandardScaler()
            z = z_scaler.fit_transform(z.cpu().numpy())
            
            det_cov = []
            
            for yi in range(args.n_clusters):
                idx = y_pred == yi
                xk = z[idx]
                detk = xk.shape[0] / data_tensor.shape[0]
                # if xk.shape[0] < 5:
                #     detk = -1
                # else:
                #     covk = np.cov(xk.T)
                #     detk = np.linalg.det(covk)
                
                det_cov.append(detk)
                
                
            det_cov_list.append(det_cov)
            # det_cov_list.append([1]*args.n_clusters)
                
            cov_str = ', '.join([f'{det_i:.2f}' for det_i in det_cov])

            pbar.set_description((
                f'Training {args.name} | '
                f'ae_loss: {ae_loss.item():.3f}; '
                f'{loss_name}: {kl_loss.item():.3f}; softmax_loss: {kl_loss2.item():.3f}; '
                # f'centroid_loss: {c1_loss.item():.3f}; gini_loss: {c2_loss.item():.3f}; '
                
                # f'total loss: {loss.item():.5f}; '
                f'acc: {accuracy:.3f}; ari: {ari:.3f}; '
                # f'delta_label: {delta_label:.5f}; delta_centroids: {delta_centroids:.5f}'
                f'det_cov: {cov_str}; '
                # f'freeze_z: {freeze_z}; '
                # f'a = {alpha}, g = {gamma}; '
                # f'w: {w}; '
                
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
    from utils.tabular_data import dataset_list
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
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='coefficient of clustering loss from softmax head')
    parser.add_argument('--update_interval', default=30, type=int,
                        help='updates target (P) at set intervals')
    parser.add_argument('--ce', action='store_true',
                        help='use CE instead of KLD/JSD')
    parser.add_argument('--nll', action='store_true',
                        help='use NLL instead of CE')
    parser.add_argument('--delay_gamma', default=0, type=int,
                        help='set gamma=0 for n epochs. default n = 0')
    parser.add_argument('--delay_alpha', default=0, type=int,
                        help='set alpha=0 for n epochs. default n = 0')
    parser.add_argument('--finetune_epochs', default=1e3, type=int)
    parser.add_argument('--early_stop', action='store_true',
                        help='early stop if clustering assignment change is below tol')
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--l_rate', default=0.001, type=float,
                        help="Learning Rate")
    parser.add_argument('--pretrain_epochs', default=1e3, type=int)
    
    parser.add_argument('--gceals_q', action='store_true',
                        help='use gceals clustering assignment Q')
    parser.add_argument('--gceals_p', action='store_true',
                        help='use gceals target P with kmeans')
    
    parser.add_argument('--gceals_p_gmm', action='store_true',
                        help='use gceals target P with kmeans')
    parser.add_argument('--sigma', action='store_true',
                        help='use learnable sigma (causes nan loss)')
    parser.add_argument('--gmm_mu', action='store_true',
                        help='initialize network mu based on sklearn GMM')
    parser.add_argument('--gmm_sigma', action='store_true',
                        help='initilize network sigma based on sklearn GMM')
    parser.add_argument('--use_svd', action='store_true',
                        help='decompose sigma with SVD')
    parser.add_argument('--dec', action='store_true',
                        help='regress to DEC')
    parser.add_argument('--device', default='cpu', type=str,
                        help="use 'cuda:0' to select cuda devices")
    parser.add_argument('--name', type=str, default='model',
                        help="model nickname)")
    parser.add_argument('--debug_autograd', action='store_true',
                        help='debug autograd')
    parser.add_argument('--temp', default=1.00, type=float)
    parser.add_argument('--eps', default=5e-3, type=float) # used
    parser.add_argument('--print', action='store_true',
                        help='print-friendly plots')
    parser.add_argument('--timed', action='store_true',
                        help='skip plots & prints to save compute cycles')
    parser.add_argument('--jsd', action='store_true',
                    help='use JSD instead of KLD')
    parser.add_argument('--depict_p', action='store_true',
                        help='replace idec_p with depict_p')
    parser.add_argument('--minibatch_p', action='store_true',
                        help='compute p on minibatch instead on update interval')
    parser.add_argument('--avg_probs', action='store_true',
                        help='take average q from cluster n softmax head')
    parser.add_argument('--test_covs', action='store_true',
                        help='do cov = I + I*v where column vector v is learned')
    parser.add_argument('--test_covs2', action='store_true',
                        help='do cov = I + I*(v^2) where column vector v is learned')
    parser.add_argument('--scalar_sigma', default=1.0, type=float,
                        help='scalar sigma if sigma is not learnable')
    parser.add_argument('--plot_comp_graph', action='store_true',
                        help='plot comp graph n exit')
    parser.add_argument('--plot_all_tsne', action='store_true',
                        help='plot tnse for all updates')
    parser.add_argument('--ce_p', action='store_true',
                        help='use idec(q) as target for ce instead of q')
    parser.add_argument('--redo_pretrain', action='store_true',
                        help='redo pretraining of AE')
    parser.add_argument('--corruptor_rate', default=0.2, type=float)
    parser.add_argument('--corruptor_type', default='pass', type=str,
                    choices=['pass', 'noise', 'sample', 'draw', 'draw-feature']) # pass, noise, sample, draw, draw-feature
    parser.add_argument('--update_full_p', action='store_true',
                        help='update_full_p every interval. true for idec')
    parser.add_argument('--reducer', default='tsne', type=str,
                    choices=['tsne', 'umap', 'pca']) # 
    parser.add_argument('--plot_dbscan', action='store_true',
                        help='plot dbscan(Z)')
    parser.add_argument('--reverse_softplus', action='store_true',
                        help='compute reverse_softplus for test_cov2 init')
    parser.add_argument('--use_w', action='store_true',
                        help='do q = q*w')
    parser.add_argument('--use_w_softmax', action='store_true',
                        help='do q_softmax = q_softmax*w')
    parser.add_argument('--w_minibatch', action='store_true',
                        help='do q = q*w where w is calculated from minibatch q')
    parser.add_argument('--centroid_loss', action='store_true',
                        help='do centroid loss')
    parser.add_argument('--gini_loss', action='store_true',
                        help='do gini loss')
    parser.add_argument('--static_centroids', action='store_true',
                        help='')
    parser.add_argument('--plot_interval', default=100, type=int,
                        help='')
    parser.add_argument('--tsne_dim', default=2, type=int,
                        help='')
    parser.add_argument('--early_stop_w', action='store_true',
                        help='')
    parser.add_argument('--norm_q', action='store_true',
                        help='')
    parser.add_argument('--update_using_wq', action='store_true',
                        help='')
    parser.add_argument('--init_on_tsne', action='store_true',
                        help='')
    parser.add_argument('--stop_w_factor', default=0.5, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    # args.dataset = int(args.dataset)
    args.pretrain_epochs = int(args.pretrain_epochs)
    args.finetune_epochs = int(args.finetune_epochs)
    # gceals_p is set to True if gceals_p_gmm is true
    args.gceals_p = args.gceals_p_gmm or args.gceals_p
    # sigma is set to True if gmm_sigma is true
    args.sigma = args.gmm_sigma or args.sigma
    # print(args)
    
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.autograd.set_detect_anomaly(args.debug_autograd)
    if args.debug_autograd:
        torch.set_printoptions(threshold=10_000)

    if args.dec:
        args.gceals_p = False
        args.gceals_q = False
        args.sigma = False

    start_time = time.time()

    X_actual, y_actual = get_data(args.dataset)
    n_classes = len(np.unique(y_actual))
    
    # print(pd.Series(y_actual).value_count())

    # use n_cluster = n_classes when 0 is given / default value
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
    
#     for yi in range(args.n_clusters):
#         idx = y_pred == yi
#         xk = z[idx]
        
#         covk = np.cov(xk.T)
#         detk = np.linalg.det(covk)

#         # print(covk)
#         # print(detk)

    from utils.pickle import save_var
    save_var(full_q.cpu().numpy(), f'./predictions/{args.name}.pkl')

    end_time = time.time()

    acc = metrics.acc(y_actual, y_pred)
    nmi = metrics.nmi(y_actual, y_pred)
    ari = metrics.ari(y_actual, y_pred)
    ch_score_manual = compute_ch_index(z, y_actual)
    
    # print('centroids')
    # print(model.centroids)
    # print('covs')
    # print(model.get_cov())

    print('Dataset:', args.dataset)
    print('n_samples:', data_tensor.shape[0])
    print('n_features:', data_tensor.shape[1])
    print('n_classes:', n_classes)
    print('n_clusters:', args.n_clusters)
    print('Time:', end_time - start_time)
    print('ACC:', acc)
    print('NMI:', nmi)
    print('ARI:', ari)
