import os,sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
import torch
import random
from scipy.stats import dirichlet
import pandas as pd
import plotly.express as px
from sklearn.utils import shuffle

class SBdata(Dataset):
    def __init__(self,samples=10000,seed=np.random.randint(20),
                    indicate=0,num_param=3,alpha=1):
        self.samples = samples
        self.seed = seed
        self.indicate=indicate
        self.num_param=num_param
        self.alpha=alpha
        np.random.seed(self.seed)
        self.beta, self.pi = self.__simulatedata__()
        
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx=0):
        return(self.beta[idx,:], self.pi[idx,:])
    
    def __simulatedata__(self):
        # Dir(alpha,alpha,alpha), alpha~Uniform(0.5,2)
        if (self.indicate==0):
            #generate beta
            beta=np.random.beta(1, self.alpha, size=(self.samples,self.num_param))
            #generate pi
            pi = np.empty_like(beta)
            pi[:, 0] = beta[:, 0]
            pi[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)
            pi[:,self.num_param-1] = 1-np.sum(pi[:,:self.num_param-1],axis=1)

            return (beta ,pi)
