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
from generate_gaussian_data import fig2data

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
        z_mu = self.mu(hidden2)
        # z_mu is of shape [batch_size, z_dim]
        z_sd = self.sd(hidden2)
        # z_sd is of shape [batch_size, z_dim]
        return z_mu, z_sd

class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self,z_dim, hidden_dim1, hidden_dim2, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.out = nn.Linear(hidden_dim1, input_dim)
    def forward(self, x):
        # x is of shape [batch_size, z_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = torch.tanh(self.linear2(hidden1))
        # hidden2 is of shape [batch_size, hidden_dim2]
        pred = self.out(hidden2)
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

def calculate_loss(x, reconstructed_x, mean, log_sd):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed_x, x, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_sd - mean.pow(2) - log_sd.exp())
    return RCL + KLD

###### intializing data and model parameters
input_dim = 200
batch_size = 500    
hidden_dim1 = 100
hidden_dim2 = 50
z_dim = 10
samples = 100000

###### creating data, model and optimizer
train_ds = fig2data(dataPoints=input_dim, samples=samples,indicate=0)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model = model.to(device)
    
###### running for 250 epochs
t = trange(50)
for e in t:
    # set training mode
    model.train()
    total_loss = 0
    for i,x in enumerate(train_dl):
        x = x[1].float().to(device)
        optimizer.zero_grad()   # zero the gradient buffers
        reconstructed_x, z_mu, z_sd = model(x) # fwd pass
        loss = calculate_loss(x, reconstructed_x, z_mu, z_sd) # loss cal
        loss.backward() # bck pass
        total_loss += loss.item() 
        optimizer.step() # update the weights
    t.set_description(f'Loss is {total_loss/(samples*input_dim):.3}')
    
 ###### Sampling 5 draws from learnt model
model.eval() # model in eval mode
z = torch.randn(5, z_dim).to(device) # random draw
with torch.no_grad():
    sampled_y = model.decoder(z)
    
df = pd.DataFrame(sampled_y)
observed_x = np.linspace(-4, 4, 200)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
for i in range(5):
    ax.plot(observed_x,df.iloc[i,:])
ax.set_xlabel('$x$',fontsize=20)
ax.set_ylabel('$y$',fontsize=20)
ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)
ax.set_title('Sampling',fontsize=30)
plt.savefig('sample_prior_vae_1d_fixed.pdf')

###### Inference on observed data
observed_x = np.linspace(-4, 4, 200)
observed_y = observed_x**3
model = model.to('cpu')
decoder_dict = model.decoder.state_dict()
y = observed_y + np.random.randn(input_dim) * 3
stan_data = {'m1': z_dim, 
             'n1': hidden_dim1,
             'n2': hidden_dim2,
             'N': input_dim,
             'W1': decoder_dict['linear1.weight'].T.numpy(),
             'b1': decoder_dict['linear1.bias'].T.numpy().reshape(1,hidden_dim2),
             'W2': decoder_dict['linear2.weight'].T.numpy(),
             'b2': decoder_dict['linear2.bias'].T.numpy().reshape(1,hidden_dim1),
             'W3': decoder_dict['out.weight'].T.numpy(),
             'b3': decoder_dict['out.bias'].T.numpy().reshape(1,input_dim),
             'y':y,
             'cov': np.identity(z_dim),
             'mu': np.array([0]*z_dim)}
             
sm = pystan.StanModel(file='vae_gaussian.stan')
fit = sm.sampling(data=stan_data, iter=10000, warmup=500, chains=3)
out = fit.extract(permuted=True)
print(fit)

df = pd.DataFrame(out['y2'])
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot(observed_x, observed_x**3, color='black',label="True Function")
ax.scatter(observed_x, y.reshape(-1,1), s=10, color='black',label="Observations")
ax.plot(observed_x, df.quantile(0.025).to_numpy().reshape(-1,1), color='slategrey', alpha=0.5)
ax.plot(observed_x, df.quantile(0.975).to_numpy().reshape(-1,1), color='slategrey', alpha=0.5)
ax.fill_between(observed_x, df.quantile(0.025).to_numpy().reshape(200,), 
                df.quantile(0.975).to_numpy().reshape(200,),color="blue", 
                alpha=.3,label="95% Confidence interval")
ax.plot(observed_x, df.mean().to_numpy().reshape(-1,1), color='red',label="Fitted Function")
ax.set_xlabel('$x$',fontsize=20)
ax.set_ylabel('$y$',fontsize=20)
ax.set_title('Inference Fit',fontsize=30)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
ax.legend(fontsize=15)
plt.savefig('inference_vae_fig2_1.pdf')

