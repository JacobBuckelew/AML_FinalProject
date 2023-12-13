import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math


class BatchNormLayer(nn.Module):
    """
    Batch Normalization accelerates training by reducing covariate shifts between layers.
    Code is adapted from the original minibatch norm algorithm in [Ioffe and Szegedy (2015)]
    as well as the moving average modification used in [Dinh et. al (2016)]
    @param eps: Hyperparameter ensuring numerical stability
    @param momentum: Controls the update of the moving average
    """
    def __init__(self, num_features, momentum=.95, eps=1e-5):
        super(BatchNormLayer, self).__init__()
        self.momentum = momentum
        self.eps = eps

        # Beta and Gamma will be model parameters
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))

        # Define non-trainable running averages of variance and mean
        # we can update the in-place
        self.register_buffer('run_m', torch.zeros(num_features))
        self.register_buffer('run_v', torch.ones(num_features))

    """
    Forward pass through batch normalization layer
    @param x: torch tensor
    @return: normalized tensor and a scalar log determinant of jacobian
    """
    def forward(self, x, cond=None):
        # original minibatch algorithm

        if self.training:
            # calculate mean and variance of minibatch
            self.b_mean = x.mean(0)
            self.b_var = x.var(0)

            # update running mean and variance in-place (Dinh)
            self.run_m.mul_(self.momentum).add_((1 - self.momentum) * self.b_mean.data)
            self.run_v.mul_(self.momentum).add_((1- self.momentum) * self.b_var.data)
            var = self.b_var
            mean = self.b_mean
        else:
            mean, var = self.run_m, self.run_v
        
        # Normalize input and scale + shift using gamma and beta

        y = self.gamma.exp() * ((x - mean) / torch.sqrt(var + self.eps)) + self.beta

        # Compute log determinant of jacobian (Dinh)
        log_det = self.gamma - (0.5 * torch.log(self.eps + var))
        return y, log_det.expand_as(x)
    
    def inverse(self, y, cond=None):
        if self.training:
            mean = self.b_mean
            variance = self.b_var
        else:
            mean = self.run_m
            variance = self.run_v

        x = (y - self.beta) * torch.exp(-self.gamma) * torch.sqrt(variance + self.eps) + mean

        log_det = 0.5 * (torch.log(variance + self.eps) - self.gamma)

        return x, log_det.expand_as(x)

class CouplingLayer(nn.Module):
    def __init__(self, n_features, mask, st_units, cond_size=None):
        super(CouplingLayer, self).__init__()
        self.register_buffer('mask', mask)
        self.snet = []
        self.tnet = []

        # build the snet (scaling function)
        if cond_size:
            self.snet.append(nn.Linear(n_features + cond_size, st_units[0]))
        else:
            self.snet.append(nn.Linear(n_features, st_units[0]))
        for units in st_units:
            self.snet.append(nn.Tanh())
            self.snet.append(nn.Linear(units, units))
        self.snet.append(nn.Tanh())
        self.snet.append(nn.Linear(st_units[0], n_features))
        self.snet = nn.Sequential(*self.snet)

        # build tnet
        if cond_size:
            self.tnet.append(nn.Linear(n_features + cond_size, st_units[0]))
        else:
            self.tnet.append(nn.Linear(n_features, st_units[0]))
        for units in st_units:
            self.tnet.append(nn.ReLU())
            self.tnet.append(nn.Linear(units, units))
        self.tnet.append(nn.ReLU())
        self.tnet.append(nn.Linear(st_units[0], n_features))
        self.tnet = nn.Sequential(*self.tnet)

    def conditioner(self, x, condition):
        if condition is not None:
            x = torch.concat((x, condition), axis= -1)
        # perform affine coupling on x_[1:d]
        # scale function
        scale_x = x
        for net in self.snet:
            scale_x = net(scale_x)
        scale = scale_x
        # translation function
        translate_x = x
        for net in self.tnet:
            translate_x = net(translate_x)
        shift = translate_x
        return scale, shift
    
    # Perform density estimation: real-world dist. ---> base dist.
    def forward(self, x, condition):
        x_mask = x * self.mask
        reverse_mask = 1 - self.mask
        # use our neural networks for approximating s() and t()
        scale, translate = self.conditioner(x_mask, condition)
        # RealNVP using Masking to partition the data
        y = x_mask + (reverse_mask * ((x - translate) * torch.exp(-1 * scale)))

        # update the log absolute det of Jacobian
        det_jacobian = -1 * reverse_mask * scale
        return y, det_jacobian

    # Sampling: base dist. --> real-world dist.
    def inverse(self, y, conditions):
        y_mask = y * self.mask
        reverse_mask = 1 - self.mask
        scale, translate = self.conditioner(y_mask, conditions)
        x = y_mask + (reverse_mask * (y * torch.exp(scale) + translate))

        det_jacobian = reverse_mask * scale
        return x, det_jacobian


class RealNVP(nn.Module):
    def __init__(self, 
                 num_features, 
                 st_units, 
                 st_layers, 
                 cond_size,
                 num_blocks, 
                 b_norm= True, 
                 momentum=0.95, 
                 learning_rate=1e-3):
        
        super().__init__()

        # define a prior (base) distribution ~ Standard Normal
        self.register_buffer('base_mean', torch.zeros(num_features))
        self.register_buffer('base_var', torch.ones(num_features))
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.b_norm = b_norm
        self.num_blocks = num_blocks
        self.cond_size = cond_size
        self.st_units = [st_units] * st_layers

        # define a mask

        self.mask = torch.arange(num_features).float() % 2

        # build model
        self.flow = []
        for i in range(num_blocks):
            # append coupling layers
            self.flow += [CouplingLayer(n_features = num_features, mask = self.mask, st_units =self.st_units, cond_size=self.cond_size)]
            # batch norm
            if self.b_norm:
                self.flow += [BatchNormLayer(num_features =num_features, momentum=momentum)]
            self.mask = 1 - self.mask

        # define as parameter for optimization
        self.flow = nn.Sequential(*self.flow)

    @property        
    def prior(self):
        return D.Normal(self.base_mean, self.base_var)
    
    # Inference Direction
    def forward(self, x, y=None):
        # sum up log det of jacobian as data passes through layers
        sum_ldj = 0
        for module in self.flow:
            x, ldj = module(x,y)
            sum_ldj = sum_ldj + ldj
        return x, sum_ldj
    
    # Sampling direction
    def inverse(self, z, y=None):
        sum_ldj = 0
        for module in self.flow[::-1]:
            z, ldj = module(z,y)
            sum_ldj = sum_ldj + ldj
        return z, sum_ldj
    
    # Loss function
    def log_density(self, x, y=None):
        z, sum_ldj = self.forward(x, y)
        #print("z:", z)
        # Log Likelihood formula using change of variables technique
        log_likelihood = torch.sum(self.prior.log_prob(z) + sum_ldj, dim=1)
        #log_likelihood = self.prior.log_prob(z) + sum_ldj
        return log_likelihood


