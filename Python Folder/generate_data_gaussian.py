import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange
import pystan
import pandas as pd
import os,sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import random

class fig2data(Dataset):
    def __init__(self, dataPoints=20, samples=10000,
                        seed=np.random.randint(20),indicate=0):
        self.dataPoints = dataPoints
        self.samples = samples
        self.seed = seed
        self.Max_Points = samples * dataPoints
        self.indicate=indicate
        np.random.seed(self.seed)
        self.evalPoints, self.data = self.__simulatedata__()
        
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx=0):
        return(self.evalPoints[:,idx], self.data[:,idx])
    
    def __simulatedata__(self):
        
        def cubic(x):
            a=random.uniform(0, 1)
            b=random.uniform(0, 1)
            c=random.uniform(0, 1)
            d=random.uniform(0, 1)
            return a*x**3+b*x**2+c*x+d
        
        if (self.indicate==0):
            X_ = np.linspace(-4, 4, self.dataPoints)
            y_samples = np.zeros((self.dataPoints,self.samples))
            for idx in range(self.samples):
                y_samples[:,idx]=cubic(X_[:, np.newaxis]).reshape(self.dataPoints,)
            return (X_.repeat(self.samples).reshape(X_.shape[0],self.samples) ,
                        y_samples)
        
        if (self.indicate==1):
            X_ = np.linspace(-4, 4, self.Max_Points)
            X_ = np.random.choice(X_, (self.dataPoints,self.samples))
            X_.sort(axis=0)
            y_samples = np.zeros((self.dataPoints,self.samples))
            for idx in range(self.samples):
                x_ = X_[:,idx]
                y_samples[:,idx] = cubic(x_[:, np.newaxis]).reshape(self.dataPoints,) 
            return (X_, y_samples)
        
        if (self.indicate==2):
            X_ = np.random.uniform(-4,4,self.dataPoints)
            y_samples = np.zeros((self.dataPoints,self.samples))
            for idx in range(self.samples):
                y_samples[:,idx]=cubic(X_[:, np.newaxis]).reshape(self.dataPoints,)
            return (X_.repeat(self.samples).reshape(X_.shape[0],self.samples) ,
                        y_samples)
        
        if (self.indicate==3):
            X_ = np.random.uniform(-4, 4, self.Max_Points)
            X_ = np.random.choice(X_, (self.dataPoints,self.samples))
            X_.sort(axis=0)
            y_samples = np.zeros((self.dataPoints,self.samples))
            for idx in range(self.samples):
                x_ = X_[:,idx]
                y_samples[:,idx] = cubic(x_[:, np.newaxis]).reshape(self.dataPoints,) 
            return (X_, y_samples)
            
import seaborn as sns
train_ds = fig2data(dataPoints=200, samples=5,indicate=0)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
observed_x = np.linspace(-4, 4, 200)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
sns.set()
for i,x in train_dl:
    ax.plot(observed_x, x.numpy().reshape(200,))
ax.set_xlabel('$x$',fontsize=20)
ax.set_ylabel('$y$',fontsize=20)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
ax.set_title('Training Dataset',fontsize=30)
plt.savefig('train_vae_1d_fixed.pdf')

