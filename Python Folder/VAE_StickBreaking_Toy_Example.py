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
from Stick_Breaking_Data import SBdata
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
import seaborn as sns
import scipy.stats
import pymc3 as pm
from scipy.stats import multivariate_normal
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.alpha = nn.Linear(hidden_dim2, z_dim)
        self.beta = nn.Linear(hidden_dim2, z_dim)
        self.softplus1 = torch.nn.Softplus(beta=1, threshold=20)
    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden1 = torch.tanh(self.linear1(x))
        # hidden1 is of shape [batch_size, hidden_dim1]
        hidden2 = self.linear2(hidden1)
        # hidden2 is of shape [batch_size, hidden_dim2]
        alpha = self.alpha(hidden2)
        beta = self.beta(hidden2)
        # z_mu is of shape [batch_size, z_dim]
        alpha = self.softplus1(alpha)
        beta=self.softplus1(beta)
        # z_sd is of shape [batch_size, z_dim]
        return alpha+1e-7,beta+1e-7

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
        out1 = self.out1(hidden2)
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

    def reparameterize(self, alpha,beta):
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        # sample from the dirichlet distribution having latent parameters alpha
            # reparameterize
        u = torch.rand(alpha.shape)
        v = (1-u.pow(1/beta)).pow(1/alpha)
        return v

    def forward(self, x):
        # encode
        alpha,beta = self.encoder(x)
        # reparameterize
        x_sample = self.reparameterize(alpha,beta)
        # decode
        generated_x = self.decoder(x_sample)

        return generated_x, alpha,beta

def calculate_loss(reconstructed1,target, alpha,beta,alpha0,beta0):
    RCL = F.mse_loss(reconstructed1, target, reduction='sum')
    kld=torch.zeros((alpha.shape[0],alpha.shape[1]))
    for i in range(100):
        kld=kld+torch.mul(1/(i+1+torch.mul(alpha,beta)),torch.exp(torch.lgamma((i+1)/alpha+1e-7)+torch.lgamma(beta)-torch.lgamma((i+1)/alpha+beta+1e-7)))

    KLD = torch.mul(torch.mul(alpha-alpha0,1/alpha),-0.57721-torch.digamma(beta)-1/beta)+torch.log(torch.mul(alpha,beta)+1e-7)+math.lgamma(alpha0+1e-7)+math.lgamma(beta0+1e-7)-math.lgamma(alpha0+beta0+1e-7)-torch.mul(beta-1,1/beta)+(beta0-1)*torch.mul(beta,kld)
    KLD = torch.sum(KLD)/50
    return RCL , KLD

if __name__ == '__main__':
    ###### intializing data and model parameters
    batch_size = 50
    hidden_dim1 = 20
    hidden_dim2 = 15
    z_dim = 9
    samples = 10000
    num_param=5
    input_dim =5
    alpha0=1
    beta0=2

    model = VAE(input_dim, hidden_dim1, hidden_dim2, z_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = model.to(device)
    
    ###### creating data
    ds = SBdata(samples=samples, indicate=0,num_param=num_param,alpha=0.5)
    train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    ###### train
    t = trange(20)
    for e in t:
        model.train()
        total_loss = 0
        for i,x in enumerate(train_dl):
            
            #input for VAE (flattened)
            x_ = x[1].float().to(device)#.sort(descending=True)[0]
            #make gradient to be zero in each loop
            optimizer.zero_grad()
            #get output
            reconstructed_x, alpha,beta = model(x_)
            #change dimensionality for computing loss function
            reconstructed_x1=reconstructed_x.reshape(batch_size,1,-1)[:,0]
            #loss 
            rcl,kdl=calculate_loss(reconstructed_x1,x_,alpha,beta,alpha0,beta0)
            #print(kdl)
            #print(rcl,kdl)
            loss=rcl+kdl
            #compute gradient
            loss.backward() 
            #if gradient is nan, change to 0
            for param in model.parameters():
                    param.grad[param.grad!=param.grad]=0
                #print(param)
            #add to toal loss
            total_loss += loss.item()
            optimizer.step() # update the weigh
        t.set_description(f'Loss is {total_loss/(samples):.3}')

    ###### Sampling 5 draws from learnt model
    model.eval() # model in eval mode
    z = torch.randn(10, z_dim).to(device) # random draw
    with torch.no_grad():
        sampled_y = model.decoder(z)
    print(sampled_y)
    
    ds1 = SBdata(samples=1, indicate=0,num_param=5,alpha=0.5,seed=6829)
    test_dl = DataLoader(ds1, batch_size=1, shuffle=True)
    #true pi
    for no,df in enumerate(test_dl):
        pi=df[1]
    print(pi)
    #true theta mu star
    np.random.seed(30)
    theta_mu_star=np.random.normal(0,2,5)
    print(theta_mu_star)
    theta_sd_star=0.7
    random.seed(50)
    theta=random.choices(np.array(theta_mu_star), weights=np.array(pi)[0],k=270)
    #sample from true model
    rv=multivariate_normal(theta,0.49*np.identity(270),1)
    np.random.seed(123)
    x=rv.rvs(size=1)
    x=x.reshape(270,)  
    #plot true probability 
    x_new = np.linspace(-6, 6, 300, endpoint=False)
    true_pdf = scipy.stats.norm.pdf(x_new,theta_mu_star[:,np.newaxis],0.7)
    true_pdfs = np.matmul(pi , true_pdf).reshape(300,)
    #print(true_pdfs)

    #use stan
    model = model.to('cpu')
    decoder_dict = model.decoder.state_dict()
    
    stan_data = {'K':input_dim,
                'n1':hidden_dim2,    
                'n2':hidden_dim1,
                'm1':z_dim,
                'N': 270,
                'M':300,
                'x':x,
                'W1': decoder_dict['linear1.weight'].T.numpy(),
                'b1': decoder_dict['linear1.bias'].T.numpy().reshape(1,hidden_dim2),
                'W2': decoder_dict['linear2.weight'].T.numpy(),
                'b2': decoder_dict['linear2.bias'].T.numpy().reshape(1,hidden_dim1),
                'W3': decoder_dict['out1.weight'].T.numpy(),
                'b3': decoder_dict['out1.bias'].T.numpy().reshape(1,input_dim),
                'cov1': np.identity(input_dim)*4,
                'mu1': np.array([0]*input_dim)}
        
    #### stan code
    sm = pystan.StanModel(file='VAE_SB_Stan_toy_example_beta.stan')
    fit = sm.sampling(data=stan_data, iter=10000, warmup=500, chains=4)
    out = fit.extract(permuted=True)
    print(fit)
    
    post_pdf_contribs = scipy.stats.norm.pdf(np.atleast_3d(x_new),
                                      out['theta_mu_star'][:, np.newaxis, :],0.7)
    post_pdfs = (out['pi'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    plt.xlim(-6,6)
    plt.hist(x, bins=20, normed=True, alpha=0.5,histtype='stepfilled', color='steelblue',edgecolor='none')
    sns.lineplot(x_new,true_pdfs,ax=ax,color="black",label="True Density")
    sns.lineplot(x_new,post_pdfs.mean(axis=0),ax=ax,color="red",label="VAE Estimation")
    ax.plot(x, [0.01]*len(x), '|', color='k')
    ax.fill_between(x_new, post_pdf_low, post_pdf_high,color="blue", 
                    alpha=.3,label="95% Confidence Interval")
    ax.set_xlabel('x',fontsize=20)
    ax.set_ylabel('Density',fontsize=20)
    ax.set_title('Inference Fit',fontsize=30)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.legend(fontsize=15,loc="upper right")
    plt.savefig("Inference Fit for Toy Example (Beta).png")
    
    df1 = pd.DataFrame(out['pi'])

    fig1 = plt.figure(figsize=(40,30),facecolor='white')
    for i in range(1, 6):
        ax = fig1.add_subplot(5, 1, i)
        ax.plot(np.linspace(1,9500,9500,endpoint=True),np.array(df1)[:9500,i-1],alpha=0.5,label="Chain 1")
        ax.plot(np.linspace(1,9500,9500,endpoint=True),np.array(df1)[9500:19000,i-1],alpha=0.5,label="Chain 2")
        ax.plot(np.linspace(1,9500,9500,endpoint=True),np.array(df1)[19000:28500,i-1],alpha=0.5,label="Chain 3")
        plt.title("$\\pi$"+str(i),fontsize=50)
        plt.ylabel("$\\pi$"+str(i),fontsize=45)
        plt.xlabel("Iterations",fontsize=45)
        ax.xaxis.set_tick_params(labelsize=40)
        ax.yaxis.set_tick_params(labelsize=40)
        ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig("VAE Traceplot Toy Example (Beta).png")
    
    import matplotlib.ticker as mtick
    #rhat
    a_=fit.summary()['summary_colnames'].index("Rhat")

    data_1=fit.summary('Z')["summary"][:, a_].reshape(1,z_dim)
    data_2=fit.summary('theta_mu_star')["summary"][:, a_].reshape(1,input_dim)
    data_3=fit.summary('pi')["summary"][:, a_].reshape(1,input_dim)

    df_1=pd.DataFrame(data_1)
    df_2=pd.DataFrame(data_2)
    df_3=pd.DataFrame(data_3)

    df_=pd.concat([df_1, df_2.reindex(df_1.index),df_3.reindex(df_1.index)], axis=1)

    name = []
    for i in range(z_dim):
        name.append('Z'+str(i+1))
    for i in range (input_dim):
        name.append('mu'+str(i+1))
    for i in range (input_dim):
        name.append('pi'+str(i+1))
    np.array(name).reshape(1,z_dim+2*input_dim)

    df_.columns = name

    fig, ax = plt.subplots(figsize=(10,10))
    y_pos = np.arange(len(name))
    Rhat = df_.iloc[0]
    plt.barh(y_pos, Rhat-1, align='center',height=0.3)
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: x+1))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(name,fontsize=15)
    ax.invert_yaxis()
    plt.axvline(x= 0 ,color="red")
    ax.xaxis.set_tick_params(labelsize=15)
    ax.xaxis.set_ticks(np.arange(-1,15,5))
    ax.set_xlabel('$\hat{R}$', fontsize = 15)
    ax.set_title('$\hat{R}$ of MCMC', fontsize = 20)
    plt.savefig("VAE Rhat Toy Example (Beta).png")
    
    a=fit.summary()['summary_colnames'].index("n_eff")

    data1=fit.summary('Z')["summary"][:, a].reshape(1,z_dim)
    data2=fit.summary('theta_mu_star')["summary"][:, a].reshape(1,input_dim)
    data3=fit.summary('pi')["summary"][:, a].reshape(1,input_dim)

    df1=pd.DataFrame(data1)
    df2=pd.DataFrame(data2)
    df3=pd.DataFrame(data3)

    df=pd.concat([df1, df2.reindex(df1.index),df3.reindex(df1.index)], axis=1)
        #N_eff/N
    df=df/30000
    #change column name
    name = []
    for i in range(z_dim):
        name.append('Z'+str(i+1))
    for i in range (input_dim):
        name.append('mu'+str(i+1))
    for i in range (input_dim):
        name.append('pi'+str(i+1))
    np.array(name).reshape(1,z_dim+2*input_dim)

    df.columns = name

    fig, ax = plt.subplots(figsize=(10,10))
    y_pos = np.arange(len(name))
    n_eff = df.iloc[0]
    plt.barh(y_pos, n_eff, align='center',height=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(name,fontsize=15)
    ax.invert_yaxis()
    ax.xaxis.set_tick_params(labelsize=15)
    plt.axvline(x= 1 ,color="red")
    ax.set_xlabel('$\mathregular{N_{eff}}$/N ', fontsize = 15)
    ax.set_title('Ratio between Effective Sample Size and Sample Size', fontsize = 20)
    plt.savefig("VAE Effective Sample Size Toy Example (Beta).png")
    
