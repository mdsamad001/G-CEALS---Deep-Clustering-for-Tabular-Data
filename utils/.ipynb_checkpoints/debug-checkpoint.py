import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12})
import math
from utils import metrics
from functools import partial

def generate_curves(ae_loss_list, kl_loss_list, softmax_loss_list, acc_list, det_cov_list, args=None, 
                    d_mu_steps = None,
                    det_sigma_steps = None,
                    w_steps = None,
                   ):
    # loss curve

    filename = args.name
    cluster_loss_name = 'CE'

    create_fig = partial(plt.subplots, figsize=(10, 7)
                         if not args.print else (6, 4))
    
    softmax_loss_title = r'Loss on softmax head; $\gamma$ =' + f'{args.gamma}'

    # acc curve
    f3, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 16))
    i = 0
    ax = axes[i]
    
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('W')

    det_cov_list = np.array(det_cov_list)
    update_epochs = [args.update_interval*i for i in range(len(acc_list))]
    for j in range(args.n_clusters):
        ax.plot(range(len(det_cov_list)), det_cov_list[:,j], label=f"$W_{j}$")
    ax.set_title("W changes")
    # ax.set_ylim(0, 1e-4)
    ax.legend()
    
    i += 1
    ax = axes[i]
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')

    update_epochs = [args.update_interval*i for i in range(len(acc_list))]
    ax.plot(range(len(softmax_loss_list)), softmax_loss_list,
             'g', label="$L_{cluster}$")
    
    ax.set_title(softmax_loss_title)
    ax.legend()
    
    i += 1
    ax = axes[i]
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')

    update_epochs = [args.update_interval*i for i in range(len(acc_list))]
    ax.plot(update_epochs, acc_list, 'r', label="Accuracy, $ACC$", marker='o', ms=3)
    ax.set_title(f'Accuracy over update_intervals = {args.update_interval}')
    ax.legend()
    
    plt.close()
    
    # MSE loss curves only
    f2, ax = create_fig()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.plot(range(len(ae_loss_list)), ae_loss_list, 'r',
             label="MSE loss, $L_{recon}$")
    # plt.plot(range(len(softmax_loss_list)), softmax_loss_list,
    #          'g', label="CE loss, $L_{cluster}$")
    if args.print:
        plt.ylim(0,0.4)
    plt.legend()
    plt.close()
    
    f1, ax = create_fig()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    ax.plot(update_epochs, acc_list, 'r', label="Accuracy, $ACC$")
    plt.legend()
    plt.close()

    # title page
    f0, ax = plt.subplots(figsize=(10, 3))
    plt.axis('off')

    plt.text(0.5, 0.5, args, ha='center', va='center', wrap=True)
    plt.close()
    
    
    # MSE and CE loss
    f2, ax = create_fig()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.plot(range(len(ae_loss_list)), ae_loss_list, 'g',
             label="Reconstruction loss, $L_{recon}$")
    plt.plot(range(len(softmax_loss_list)), softmax_loss_list, 'r',
             label="Cluster loss, $L_{cluster}$")
    if args.print:
        plt.ylim(0,0.4)
    plt.legend()
    plt.close()
    
    fig_list = [f0, f1, f2, f3]
    
    if d_mu_steps!=None:
        f_temp, ax = plt.subplots(figsize=(4,3))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r"$\Vert \mu^t_j - \mu^{t-1}_{j} \Vert^{2}_{2}$")

        y_list = np.array(d_mu_steps)
        x_list = range(len(y_list))
        for j in range(args.n_clusters):
            ax.plot(x_list, y_list[:,j], label=fr"$\mu_{j}$")
        # ax.set_title("Centroid changes")
        # ax.set_ylim(0, 1e-4)
        ax.legend()
        fig_list.append(f_temp)
        plt.close()
        
    if det_sigma_steps!=None:
        f_temp, ax = plt.subplots(figsize=(4,3))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'Determinant of $\Sigma_j$')

        y_list = np.array(det_sigma_steps)
        x_list = range(len(y_list))
        for j in range(args.n_clusters):
            ax.plot(x_list, y_list[:,j], label=fr"$\Sigma_{j}$")
        # ax.set_title("Sigma changes")
        # ax.set_ylim(0, 1e-4)
        ax.legend()
        fig_list.append(f_temp)
        plt.close()
        
    if w_steps!=None:
        f_temp, ax = plt.subplots(figsize=(4,3))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'Cluster weights, $\omega_j$')

        y_list = np.array(w_steps)
        x_list = range(len(y_list))
        for j in range(args.n_clusters):
            ax.plot(x_list, y_list[:,j], label=fr"$\omega_{j}$")
        # ax.set_title("W changes")
        # ax.set_ylim(0, 1e-4)
        ax.legend()
        fig_list.append(f_temp)
        plt.close()

    import matplotlib.backends.backend_pdf
    with matplotlib.backends.backend_pdf.PdfPages(f"./debug/{filename}_curves.pdf") as pdf:
        for fig in fig_list:  # will open an empty extra figure :(
            fig.tight_layout()
            pdf.savefig(fig)

def plot_tsnes(data_numpy, full_z_list, y_pred_list, y_actual, args, update_epoch_list=[]):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    

    figs = []
    c = [f'C{i}' for i in y_actual]

    kmeans = KMeans(n_clusters=args.n_clusters, init="k-means++", n_init='auto', random_state=42)
    y_pred = kmeans.fit_predict(data_numpy)
    f = get_tsne_figure(data_numpy, y_pred, y_actual, args, title=f"dataset={args.dataset}; {args.reducer}(X) + kmeans")
    figs.append(f)
    
    Z = full_z_list[0]
    y_pred = kmeans.fit_predict(Z)
    f = get_tsne_figure(Z, y_pred, y_actual, args, title=f"dataset={args.dataset}; {args.reducer}(Z) + kmeans")
    figs.append(f)

    for i, Z in enumerate(full_z_list):
        y_pred = y_pred_list[i]
        update_str = f"epoch={update_epoch_list[i]}" if len(update_epoch_list) > 0 else f"update: {i}"
        f = get_tsne_figure(Z, y_pred, y_actual, args, title=f"dataset={args.dataset}; {args.reducer}(Z) + DL at {update_str}")
        figs.append(f)

    import matplotlib.backends.backend_pdf
    filename = args.name
    with matplotlib.backends.backend_pdf.PdfPages(f"./debug/{filename}_tsne.pdf") as pdf:
        for fig in figs:  # will open an empty extra figure :(
            fig.tight_layout()
            pdf.savefig(fig)

def get_tsne_figure(data_numpy, y_pred, y_actual, args, title='TSNE plot'):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    
    reducer_map = {
        'umap': umap.UMAP(n_components=2),
        'tsne': TSNE(n_components=2, learning_rate='auto', init='random'),
        'pca': PCA(n_components=2),
    }
    
    reducer =  reducer_map[args.reducer]

    acc = metrics.acc(y_actual, y_pred)
    nmi = metrics.nmi(y_actual, y_pred)
    ari = metrics.ari(y_actual, y_pred)

    Z_tsne = reducer.fit_transform(data_numpy)
    
    create_fig = partial(plt.subplots, figsize=(10, 7)
                        if not args.print else (6, 4))
    
    f, ax = create_fig()
    plt.xlabel(f'{args.reducer}_0', fontsize=20 if args.print else 10)
    plt.ylabel(f'{args.reducer}_1', fontsize=20 if args.print else 10)

    if not args.print:
        for y_i in np.unique(y_pred):
            idx = y_pred == y_i
            plt.scatter(Z_tsne[idx, 0], Z_tsne[idx, 1], ec=f'C{y_i}', label = f'y_hat = {y_i}', fc=f'none', alpha=.85, s = 60)
            plt.scatter(np.mean(Z_tsne[idx,0]), np.mean(Z_tsne[idx,1]), ec = 'k', fc=f'C{y_i}', marker='o', s = 40)

    for y_i in np.unique(y_actual):
        idx = y_actual == y_i
        plt.scatter(Z_tsne[idx, 0], Z_tsne[idx, 1], c=f'C{y_i}', label = f'Y = {y_i}', marker='x', s=10)
    
    
        
    plt.legend()
    if not args.print:
        plt.title(f'{title}; ACC: {acc:.3f}, NMI: {nmi:.3f}, ARI: {ari:.3f}')
    plt.close()
    return f