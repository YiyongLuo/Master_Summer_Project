#vae7+stan
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
from Dirdata import Dirdata
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
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, z_dim)
        self.softplus1 = torch.nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        alpha = self.linear3(hidden2)
        # make outputs be positive
        alpha = self.softplus1(alpha)
        # alpha is of shape [batch_size, z_dim]
        return alpha

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self,z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear3 = nn.Linear(hidden_dim1, input_dim)
        #self.out2 = nn.Softmax(dim=2)
        self.softplus2 = torch.nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        # x is of shape [batch_size, z_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim2]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim1]
        out1 = torch.tanh(self.linear3(hidden2))
        # ensure outputs are positive
        out1 = self.softplus2(out1)
        # pred is of shape [batch_size, input_dim]
        return out1

class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim,beta):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim1, hidden_dim2, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim1, hidden_dim2, input_dim)
        self.beta = beta

    def reparameterize(self, alpha,beta):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        # sample from the dirichlet distribution having latent parameters alpha
            # reparameterize
        u = torch.rand(alpha.shape)
        v = 1/beta*((torch.mul(torch.mul(u,alpha),torch.exp(torch.lgamma(alpha)))).pow(1/alpha))
        v = torch.mul(v,1/torch.transpose(torch.sum(v,1).repeat(alpha.shape[1],1),0,1))
        return v

    def reparameterize1(self,x,beta,dataPoints,num_param):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        # sample from the distribution having parameter outputs
            # reparameterize
        u = torch.rand((x.shape[0],dataPoints,num_param))
        x = x.reshape((x.shape[0],dataPoints,num_param))
        y = torch.zeros((x.shape[0],dataPoints,num_param))
        for idx in range(x.shape[0]):
            v = 1/beta*((torch.mul(torch.mul(u[idx],x[idx]),torch.exp(torch.lgamma(x[idx])))).pow(1/x[idx]))
            y[idx]=torch.mul(v,1/torch.transpose(torch.sum(v,1).repeat(x[idx].shape[1],1),0,1))
        return y

    def forward(self, x,beta,dataPoints,num_param):
        # encode
        alpha = self.encoder(x)
        # reparameterize
        v = self.reparameterize(alpha,beta)
        # decode
        generated_x = self.decoder(v)
        pred = self.reparameterize1(generated_x,beta,dataPoints,num_param)
        return pred, alpha, v,generated_x

def calculate_loss(reconstructed1,target, alpha):
    RCL = F.mse_loss(reconstructed1, target, reduction='sum')
    KLD=torch.sum(torch.lgamma(alpha),1)-torch.sum(torch.tensor(([math.lgamma(1-1/alpha.shape[1])])).repeat(alpha.shape[0]*alpha.shape[1]).reshape(alpha.shape[0],alpha.shape[1]),1)+torch.sum((torch.tensor(([math.lgamma(1-1/alpha.shape[1])])).repeat(alpha.shape[0]*alpha.shape[1]).reshape(alpha.shape[0],alpha.shape[1])-alpha)*torch.digamma(torch.tensor([1-1/alpha.shape[1]])),1)
    KLD = torch.sum(KLD)
    return RCL , KLD

if __name__ == '__main__':
    ###### intializing data and model parameters
    dataPoints=200
    batch_size = 50
    hidden_dim1 = 200
    hidden_dim2 = 150
    z_dim = 50
    samples = 10000
    num_param=3
    input_dim = dataPoints*num_param
    beta = 1

    model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim,beta)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = model.to(device)
    
    ###### creating data
    ds = Dirdata(dataPoints=dataPoints, samples=samples, indicate=0,num_param=num_param)
    train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    ###### train
    t = trange(20)
    for e in t:
        model.train()
        total_loss = 0
        for i,x in enumerate(train_dl):
            #input for VAE (flattened)
            x_ = x[1].float().to(device).reshape(batch_size,1,-1)[:,0]
            #make gradient to be zero in each loop
            optimizer.zero_grad()
            #get output
            reconstructed_x, alpha, v,generated_x = model(x_,beta,dataPoints,num_param)
            #change dimensionality for computing loss function
            reconstructed_x1=reconstructed_x.reshape(batch_size,1,-1)[:,0]
            #loss 
            rcl,kdl=calculate_loss(reconstructed_x1,x_,alpha)
            loss=rcl+kdl
            #compute gradient
            loss.backward() 
            #if gradient is nan, ch1ange to 0
            for param in model.parameters():
                param.grad[param.grad!=param.grad]=0
            #add to toal loss
            total_loss += loss.item()
            optimizer.step() # update the weigh
        t.set_description(f'Loss is {total_loss/(samples*dataPoints):.3}')

###### Sampling 5 draws from learnt model
    model.eval() # model in eval mode
    u_ = torch.rand(5,z_dim)
    alpha_ = torch.tensor([1-1/z_dim]).repeat(z_dim*5).reshape(5,z_dim)
    v_ = 1/beta*((torch.mul(torch.mul(u_,alpha_),torch.exp(torch.lgamma(alpha_)))).pow(1/alpha_))
    v_ = torch.mul(v_,1/torch.transpose(torch.sum(v_,1).repeat(z_dim,1),0,1))
    # random draw
    with torch.no_grad():
        sampled_y = model.decoder(v_)
        u_1 = torch.rand((sampled_y.shape[0],dataPoints,num_param))
        x_1 = sampled_y.reshape((sampled_y.shape[0],dataPoints,num_param))
        y_1 = torch.zeros((sampled_y.shape[0],dataPoints,num_param))
        for idx in range(sampled_y.shape[0]):
            v_1 = 1/beta*((torch.mul(torch.mul(u_1[idx],x_1[idx]),torch.exp(torch.lgamma(x_1[idx])))).pow(1/x_1[idx]))
            y_1[idx]=torch.mul(v_1,1/torch.transpose(torch.sum(v_1,1).repeat(x_1[idx].shape[1],1),0,1))

    for no, y in enumerate(y_1):
        #create a dataframe
        df=pd.DataFrame(y,columns=['$\\theta_1$', '$\\theta_2$', '$\\theta_3$'])
        #plot
        fig =px.scatter_ternary(df, a='$\\theta_1$', b='$\\theta_2$', c='$\\theta_3$',title="Dirichlet Distribution Visualization")
        fig.show()

    ###### Inference on observed data
    y = dirichlet.rvs(np.array([1.6,1.6,1.6]), size=dataPoints, random_state=1)
    model = model.to('cpu')
    decoder_dict = model.decoder.state_dict()
    stan_data = {'N': input_dim,
                 'n1': hidden_dim2,
                 'n2': hidden_dim1,
                 'm1': z_dim,
                 'm2': num_param,
                 'y':y.T,
                 'W1': decoder_dict['linear1.weight'].T.numpy(),
                 'b1': decoder_dict['linear1.bias'].T.numpy().reshape(1,hidden_dim2),
                 'W2': decoder_dict['linear2.weight'].T.numpy(),
                 'b2': decoder_dict['linear2.bias'].T.numpy().reshape(1,hidden_dim1),
                 'W3': decoder_dict['linear3.weight'].T.numpy(),
                 'b3': decoder_dict['linear3.bias'].T.numpy().reshape(1,input_dim),
                 'alpha':1-1/z_dim}
    #### stan code
    sm = pystan.StanModel(file='stan_dir_vae.stan')
    fit = sm.sampling(data=stan_data, iter=10000, warmup=50, chains=4)
    print(fit)
    out = fit.extract(permuted=True)
    #print(out['y2'].shape)
    a=np.sum(out['y2'],axis=0)/(out['y2'].shape[0])
    a=a.reshape(200,3)
    df1=pd.DataFrame(a,columns=['$\\phi_1$', '$\\phi_2$', '$\\phi_3$'])
    fig1 =px.scatter_ternary(df1, a='$\\phi_1$', b='$\\phi_2$', c='$\\phi_3$',title="Dirichlet Distribution Inference")
    fig.show()
    