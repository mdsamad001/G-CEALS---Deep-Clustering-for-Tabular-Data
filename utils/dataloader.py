import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class EqualLoader:
    def __init__(self, x_list, batch_size=256, debug=0):
        self.x_list = x_list
        self.n = len(x_list) # no. of classes
        self.debug = debug
        self.sizes = [x.shape[0] for x in x_list]
        
        arg_sort = np.argsort(self.sizes)
        self.x_list = [x_list[i] for i in arg_sort]
        self.sizes = [self.sizes[i] for i in arg_sort] # min size now at idx=0
        self.min_size = self.sizes[0]
        
        self.min_size = min(self.sizes)# if batch_size == -1 else np.floor(batch_size / len(x_list))
        self.min_size = int(self.min_size)
        
        self.batch_size = batch_size
        self.possible_batch_size = self.n * self.min_size
        self.possible_batch_size = self.possible_batch_size if self.possible_batch_size < self.batch_size else self.batch_size
        
        self.possible_size_list = [int(self.possible_batch_size/self.n) for i in range(self.n)]
        self.possible_size_list[-1] = self.possible_batch_size - sum(self.possible_size_list[:-1])
        print(self.possible_size_list)
        
        self.debug and print('min size', self.min_size)
        
        self.n_batches = np.ceil(max(self.sizes) / min(self.sizes))
        
        self.debug and print(self.min_size, self.n_batches)
        
        self.seed = 0;
        self.iter_counter = 0;
        
        self._reset_loader()
        
        
    def _reset_loader(self):
        self.iter_counter = 0
        g = torch.Generator()
        g.manual_seed(self.seed)
        self.seed += 1
        self.dataloaders = [DataLoader(TensorDataset(x), batch_size=n, shuffle=True, generator=g) for x, n in zip(self.x_list, self.possible_size_list)]
        self.data_iter = [x.__iter__() for x in self.dataloaders]
        
    def _shuffle(self, x):
        self.iter_counter += 1
        g = torch.Generator()
        g.manual_seed(self.iter_counter)
        idx = torch.randperm(x.size()[0], generator=g)
        
        return x[idx]
        
        
    def __iter__(self):
        y_list = []
        
        for dl in self.dataloaders:
            y_batch = next(iter(dl))
            y_list += y_batch
            
         
        y =  torch.concat(y_list)
        
        # y_shuffled = self._shuffle(y)
            
        self.debug and print('iter', self.iter_counter)
        self.debug and print(y)
            
        yield y
        