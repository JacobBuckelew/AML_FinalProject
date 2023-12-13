import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from NF import RealNVP
import numpy as np


# Defining GNN

class GNN(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(GNN, self).__init__()

        # Linear weight matrices used in GANF implementation
        # weight matrix for AH_t
        self.lin_h_t = nn.Linear(input_size, hidden_dim)

        # weight matrix for H_T-1 (past dependencies)
        self.lin_past = nn.Linear(input_size, hidden_dim, bias=False)

        # weight matrix after applying ReLu activation
        self.lin_act = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, H, A):
        # A is N x N
        # H is B X N x T x D 
        D_h_t = self.lin_h_t(torch.einsum('nkld,nkj->njld', H, A))
        #D_h_t = self.lin_h_t(torch.einsum('bjd,bnn->bjd', H, A))
       #D_past = self.lin_past(H[:,:-1])
        D_past = self.lin_past(H[:,:,:-1])
        D_h_t[:,:,1:] += D_past
        #D_h_t[:,1:] += D_past
        D = self.lin_act(F.relu(D_h_t))
        return D

# Scaled Dot product attention

class SelfAttention(nn.Module):

    def __init__(self, t):
        super(SelfAttention, self).__init__()
        # Implementation is simply Scaled Dot Product Attention 
        # Similar to (Vaswani et. al, 2017), (Zhou et al., 2023)
        # Apply linear transformations using nn.Linear
        # Each weight matrix is of dimension t x t except the value weights matrix
        # Query weights
        self.W_q = nn.Linear(t,t)
        # Key weights
        self.W_k = nn.Linear(t,t)

        # Value weights
        #self.W_v = nn.Linear(s, s)

        # Softmax function to compute attentions 
        self.sm = nn.Softmax(dim=1)

        self.dp = nn.Dropout(0.1)
    
    def forward(self, x):
        # x is B x N x L x D
        shape = x.shape
        # reshape by combining last two dimensions
        #x_c = x.reshape((shape[0], shape[1], -1))
        x_c = x.reshape((shape[0], shape[1], -1))
        b, length, dim = x_c.size()
        query = self.W_q(x_c)
        key = self.W_k(x_c)
        key_t = key.view(b, dim, length)
        score = (query @ key_t)/ math.sqrt(length)
        # softmax and dropout 
        #A = self.W_v(self.dp(self.sm(score)))
        A = self.dp(self.sm(score))
        #A = self.dp(self.sm(score))
        return A



# class for defining the Context-Aware Normalizing Flows (CANF)
class CANF(nn.Module):

    def __init__(self, 
                 num_features,
                 num_entities, 
                 num_blocks, 
                 hidden_size, 
                 window_size, 
                 num_hidden, 
                 dropout=0.5, 
                 b_norm=True, 
                 rnn=True, 
                 gnn=True):
        super(CANF, self).__init__()
        # Self-Attention --> RNN --> GNN --> NF
        #self.att = SelfAttention(t=window_size)

        if (rnn == False and gnn == False):
            self.rnn = None
            self.gnn = None
            self.condition_size = None
        else:
            self.condition_size = hidden_size
        
        if rnn:
            # Input shape: num batches x T x ND
            self.rnn = nn.GRU(input_size=num_features, hidden_size=hidden_size, dropout=dropout,batch_first=True)
        # use GRU
        if gnn:
            self.gnn = GNN(input_size=hidden_size, hidden_dim=hidden_size)
            self.att = SelfAttention(t=window_size * num_features)
            #self.att = SelfAttention(t=window_size, s=num_entities)
        else:
            self.gnn = None
        
        self.num_entities = num_entities
        self.nf = RealNVP(num_features=num_features, st_units=hidden_size,cond_size=self.condition_size, st_layers=num_hidden, num_blocks=num_blocks, b_norm=b_norm)

    def forward(self, x, take_mean=True):
        density = self.estimate_density(x,take_mean)
        if take_mean:
            return density.mean()
        else:
            return density
        #print(density)
        #return density.mean()
    
    def estimate_density(self, x, take_mean=True):
        # shape of the whole batch is B x N x T x D
        batch_shape = x.shape
        # extract attention matrix
        if (self.gnn != None):
            A = self.att(x)
        # save graph
            self.A = A
        # reshape matrix into BN x T x D for input into GRU
        #x = x.reshape(batch_shape[0] * batch_shape[2], batch_shape[1])
        if(self.rnn != None):
            x = x.reshape(batch_shape[0] * batch_shape[1], batch_shape[2], batch_shape[3])
            h, _ = self.rnn(x)
        else:
            h = None
        # reshape h into B x N x T x H 
        # perform GNN operations 
        if (self.gnn != None):
            h = h.reshape((batch_shape[0], batch_shape[1], h.shape[1], h.shape[2]))
            #h = h.reshape(batch_shape[0], batch_shape[3], h.shape[1], h.shape[2])
            #print(h.shape)
            h = self.gnn(h, A)
        # reshape into BNT x H
        if(self.rnn != None and self.gnn != None):
            h = h.reshape((-1, h.shape[3]))
        elif(self.rnn != None and self.gnn == None):
            h = h.reshape((-1, h.shape[2]))
        else:
            h = None

    
        x = x.reshape((-1, batch_shape[3]))
        #h = h.reshape((-1, h.shape[2]))
        #x = x.reshape((-1, batch_shape[1]))
        #log_density = self.nf.log_density(x,h).reshape([batch_shape[0], -1])
        #print(log_density.shape)
        log_density = self.nf.log_density(x,h)
        if take_mean:
            #print(log_density.shape)
            log_density = log_density.reshape([batch_shape[0], -1])
            #print(log_density.shape)
            log_density = log_density.mean(dim=1)
            #print(log_density.shape)
        else:
            log_density = log_density.reshape([batch_shape[0], batch_shape[1], -1])
            log_density = log_density.mean(dim=2)
            #print(log_density.shape)
        
        return log_density
    
    def estimate_density_t(self, x):
        # shape of the whole batch is B x N x T x D
        batch_shape = x.shape
        # extract attention matrix
        A = self.att(x)
        # save graph
        self.A = A

        # reshape matrix into BN x T x D for input into GRU
        #x = x.reshape(batch_shape[0] * batch_shape[2], batch_shape[1])
        x = x.reshape(batch_shape[0] * batch_shape[1], batch_shape[2], batch_shape[3])
        h, _ = self.rnn(x)
        # reshape h into B x N x T x H 
        h = h.reshape((batch_shape[0], batch_shape[1], h.shape[1], h.shape[2]))
        # perform GNN operations on h
        h = self.gnn(h, A)
        # reshape into BNT x H
        h = h.reshape((-1, h.shape[3]))
        x = x.reshape((-1, batch_shape[3]))
        #h = h.reshape((-1, h.shape[2]))
        #x = x.reshape((-1, batch_shape[1]))
        log_density = self.nf.log_density(x,h)
        log_density_t = log_density.reshape(batch_shape[0],batch_shape[2], -1)

        log_density_t = log_density_t.mean(dim=2)
        return log_density



    


