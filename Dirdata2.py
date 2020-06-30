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

class Dirdata(Dataset):
    def __init__(self, dataPoints=20, samples=10000,
                        seed=np.random.randint(20),indicate=0,num_param=3):
        self.dataPoints = dataPoints
        self.samples = samples
        self.seed = seed
        self.Max_Points = samples * dataPoints
        self.indicate=indicate
        self.num_param=num_param
        np.random.seed(self.seed)
        self.evalPoints, self.data, self.data1,self.occure = self.__simulatedata__()
        
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx=0):
        return(self.evalPoints, self.data[idx],self.data1[idx],self.occure[idx])
    
    def __simulatedata__(self):
        # Dir(alpha,alpha,alpha), alpha~Uniform(0.5,2)
        if (self.indicate==0):
            #generate alpha
            alpha=np.random.uniform(0.5,2,self.samples)
            #repeat alpha
            alpha=np.array([alpha]*self.num_param).transpose()
            #initialize theta and counts (counts are only used in inference, not training)
            theta = np.zeros((self.samples,self.dataPoints,self.num_param))
            occurrence = np.zeros((self.samples, self.dataPoints,self.num_param))
            #generate theta 
            for idx in range(self.samples):
                #generate theta from dirichlet distr
                theta[idx]=np.random.dirichlet(alpha[idx,:],self.dataPoints)
            #shuffle theta
            theta =theta.reshape(self.samples*self.dataPoints,self.num_param)
            theta=shuffle(theta, random_state=0)
            theta1 = theta.reshape(self.samples,self.dataPoints,self.num_param)
            #generate counts
            for idx in range(self.samples):
                for idy in range(self.dataPoints):
                    occurrence[idx][idy,:]=np.random.multinomial(50,theta1[idx][idy,:],size=1)
            occurrence =occurrence.reshape(self.samples*self.dataPoints,self.num_param)
            return (alpha ,theta,theta1, occurrence)

        
        # Dir(1,1,0.2)
        if (self.indicate==4):
            alpha=np.array([1,1,0.2])
            #initialize theta and counts (counts are only used in inference, not training)
            theta = np.zeros((self.samples,self.dataPoints,self.num_param))
            occurrence = np.zeros((self.samples, self.dataPoints,self.num_param))
            #generate theta
            for idx in range(self.samples):
                #generate theta from dirichlet distr
                theta[idx]=np.random.dirichlet(alpha,self.dataPoints)
            #shuffle theta
            theta =theta.reshape(self.samples*self.dataPoints,self.num_param)
            theta=shuffle(theta, random_state=0)
            theta1 = theta.reshape(self.samples,self.dataPoints,self.num_param)
            #generate counts
            for idx in range(self.samples):
                for idy in range(self.dataPoints):
                    occurrence[idx][idy,:]=np.random.multinomial(50,theta1[idx][idy,:],size=1)
            occurrence=occurrence.reshape(self.samples*self.dataPoints,self.num_param)
            return (alpha ,theta, theta1,occurrence)

if __name__ == '__main__':
    ds =Dirdata(dataPoints=200, samples=1, indicate=0,num_param=3)
    dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    fig = plt.figure(figsize=(8,8))
    for no,dt in enumerate(dataloader):
        df=pd.DataFrame(dt[2].reshape(200,3),columns=['$\\theta_1$', '$\\theta_2$', '$\\theta_3$'])
        fig =px.scatter_ternary(df, a='$\\theta_1$', b='$\\theta_2$', c='$\\theta_3$',title="Dirichlet Distribution Visualization")
        fig.show()