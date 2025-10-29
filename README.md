## Deep Clustering of Tabular Data by Weighted Gaussian Distribution Learning
![proposed model](proposed-model.png)

More details on arXiv: [https://arxiv.org/abs/2301.00802](https://arxiv.org/abs/2301.00802)
## Requirements
We recommend users set up a Conda Environment using the env.yml file

## Notes
Consider tuning the follwing two hyperparameters below : 
* Stop factor (--stop_w_factor) can be data set dependent, so we recommend using a default value,  --stop_w_factor = 0.1 .
* Embedding size (--latent_dim) can also also data set depenedent, so we recommend using embedding size , --latent_dim = X.shape[1].

## Citation
Please cite this paper

Rabbani, S. B., Medri, I. v., & Samad, M. D. (2025). Deep clustering of tabular data by weighted Gaussian distribution learning. Neurocomputing, 623, 129359. https://doi.org/10.1016/j.neucom.2025.129359
