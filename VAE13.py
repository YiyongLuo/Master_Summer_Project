from scipy.stats import dirichlet as diri
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import math
from scipy.special import gamma, factorial
from tqdm import tqdm, trange
from Dirdata2 import Dirdata
import torch
from scipy.stats import multinomial
from torch.utils.data import Dataset, DataLoader
from scipy.stats import dirichlet
import os,sys
import pystan
import pystan
import pandas as pd
import warnings
import plotly.express as px
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, z_dim)
        self.sd = nn.Linear(hidden_dim2, z_dim)
    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        z_mu = torch.tanh(self.mu(hidden2))
        # z_mu is of shape [batch_size, z_dim]
        z_sd = torch.tanh(self.sd(hidden2))
        # z_sd is of shape [batch_size, z_dim]
        return z_mu, z_sd

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self,z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.out1 = nn.Linear(hidden_dim1, input_dim)
        self.out2 = nn.Softmax(dim=1)
    def forward(self, x):
        # x is of shape [batch_size, z_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim2]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim1]
        out1 = torch.tanh(self.out1(hidden2))
        #ensure sum of 3 elements to be 1
        pred = self.out2(out1)
        # pred is of shape [batch_size, input_dim]
        return pred

class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim1, hidden_dim2, input_dim)

    def reparameterize(self, z_mu, z_sd):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        if self.training:
            # sample from the distribution having latent parameters z_mu, z_sd
            # reparameterize
            std = torch.exp(z_sd / 2)
            eps = torch.randn_like(std)
            return (eps.mul(std).add_(z_mu))
        else:
            return z_mu

    def forward(self, x):
        # encode
        z_mu, z_sd = self.encoder(x)
        # reparameterize
        x_sample = self.reparameterize(z_mu, z_sd)
        # decode
        generated_x = self.decoder(x_sample)
        return generated_x, z_mu,z_sd

def calculate_loss(reconstructed1,target, mean, log_sd):
    RCL = F.mse_loss(reconstructed1, target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_sd - mean.pow(2) - log_sd.exp())
    return RCL + KLD

if __name__ == '__main__':
    ###### intializing data and model parameters
    dataPoints=200
    batch_size = 5
    hidden_dim1 = 70
    hidden_dim2 = 60
    z_dim = 50
    samples = 1000
    num_param=3
    input_dim = 3

    model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = model.to(device)
    
    ###### creating data
    ds =Dirdata(dataPoints=dataPoints, samples=samples, indicate=4,num_param=num_param)
    train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    ###### train
    t = trange(50)
    for e in t:
        model.train()
        total_loss = 0
        for i,x in enumerate(train_dl):
            #input for VAE (flattened)
            x_ = x[1].float().to(device)
            #make gradient to be zero in each loop
            optimizer.zero_grad()
            #get output
            reconstructed_x, z_mu, z_sd = model(x_)
            #change dimensionality for computing loss function
            reconstructed_x1=reconstructed_x.reshape(batch_size,1,-1)[:,0]
            
            #loss 
            loss=calculate_loss(reconstructed_x1,x_,z_mu,z_sd)
            #compute gradient
            loss.backward() 
            #if gradient is nan, change to 0
            for param in model.parameters():
                param.grad[param.grad!=param.grad]=0
                
            #add to toal loss
            total_loss += loss.item()
            optimizer.step() # update the weigh
        t.set_description(f'Loss is {total_loss/(samples*dataPoints):.3}')
    
    ###### Sampling 5 draws from learnt model
    model.eval() # model in eval mode
    z = torch.randn(200, z_dim).to(device) # random draw
    with torch.no_grad():
        sampled_y = model.decoder(z)
    df=pd.DataFrame(sampled_y,columns=['$\\theta_1$', '$\\theta_2$', '$\\theta_3$'])
    fig =px.scatter_ternary(df, a='$\\theta_1$', b='$\\theta_2$', c='$\\theta_3$',title="Dirichlet Distribution Visualization")
    fig.show()
    