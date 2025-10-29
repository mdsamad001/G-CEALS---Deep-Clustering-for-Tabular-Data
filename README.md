## Deep Clustering of Tabular Data by Weighted Gaussian Distribution Learning
![proposed model](proposed-model.png)

More details on arXiv: [https://arxiv.org/abs/2301.00802](https://arxiv.org/abs/2301.00802)
## Requirements
We recommend users set up a Conda Environment using the env.yml file

## Notes
Consider tuning the follwing two hyperparameters below : 
* Stop factor (--stop_w_factor) can be data set dependent, so we recommend using a default value,  --stop_w_factor = 0.1 .
* Embedding size (--latent_dim) can also also data set depenedent, so we recommend using embedding size , --latent_dim = X.shape[1].

## Run Command

Use the follwing format to run this experiment :  

```bash
python gceals.py --device cuda:0 --pretrain_epochs 1000 --finetune_epochs 1000 --update_interval 50 --l_rate 0.001 --gamma 0.0 --alpha 0.1  --ce --gceals_q --use_w --update_interval 1 --sigma --test_covs2 --reverse_softplus --name expt_expt-convergence-nostop_final-gceals-l=20_1510_42_09 --seed 42 --latent_dim 15 --dataset 1510
```

## Citation
Please cite this paper

Rabbani, S. B., Medri, I. v., & Samad, M. D. (2025). Deep clustering of tabular data by weighted Gaussian distribution learning. Neurocomputing, 623, 129359. https://doi.org/10.1016/j.neucom.2025.129359
