import numpy as np
import scipy
import math

import torch
import torch.nn as nn

def compute_clenshaw_curtis(N):
    """ Quadrature rule for approximating \int_{-1}^1 f(x) dx = \sum_i w[i]f(x[i]) """
    lam = np.arange(0, N + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / N)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / N
    W = np.arange(0, N + 1, 1).reshape(-1, 1)
    W[np.arange(1, N + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, N + 1, 2)] = 0
    weights = torch.tensor(lam.T @ W)
    points  = torch.tensor(np.cos(np.arange(0, N + 1, 1).reshape(-1, 1) * math.pi / N))
    return weights, points

def transform_points(weights, points, a=-1., b=1.):
    """ Transform quadrature points from domain [-1,1] to [a,b] """
    # rescale weights
    weights *= (b-a)/2.
    points  = (b-a)/2.*points + (a+b)/2.
    return weights, points
    
class OneDimMonotonePushForward():
    def __init__(self, T, eta):
        self.T   = T
        self.eta = eta
    def icdf(self, q):
        return self.T(self.eta.icdf(q))
    def sample(self, N):
        Z = self.eta.sample(shape=(N,1))
        X = self.T(Z)
    
class OneDimMonotoneWasserstein(nn.Module):
    def __init__(self, nu, T, eta, N, p, pts='CC'):
        super(OneDimMonotoneWasserstein, self).__init__()
        self.nu   = nu
        self.T    = T
        self.eta  = eta
        self.N    = N
        self.p    = p
        # define integration points
        w, q = compute_clenshaw_curtis(N)
        w, q = transform_points(w, q, a=0., b=1.)
        # remove CC end points
        self.w = w[1:-1]
        self.q = q[1:-1]
        # evalaute target CDF
        self.nu_icdf = self.nu.icdf(self.q).detach()
        # evaluate x points using inverse Gaussian CDF
        self.x = self.eta.icdf(self.q).detach()
        # remove any rows == inf
        inf_idx = torch.isinf(self.x)[:,0]
        self.x  = self.x[inf_idx == False,:]
        self.w  = self.w[inf_idx == False,:].double()
        
    def forward(self):
        """ Compute Wasserstein-p distance given N integration points"""
        # evaluate inverse CDFs
        inv_cdf1x = self.nu_icdf[:,0]
        inv_cdf2x = self.T.forward(self.x[:,0])[:,0]
        # evaluate Ix and compute inner product
        Ix = torch.abs(inv_cdf1x - inv_cdf2x).pow(self.p)
        return torch.dot(torch.squeeze(self.w),torch.squeeze(Ix)).pow(1./self.p)

    def optimize(self):
        if self.p==2:
            Hx = self.T.eval_basis(self.x[:,0])
            D = np.diag(self.w[:,0])
            coeff = np.linalg.solve(np.dot(np.dot(Hx.T, D), Hx), np.dot(np.dot(Hx.T, D), self.nu_icdf))
            self.T.linear.weight.data = torch.tensor(coeff).T
        else:
            raise ValueError('Use gradient-based scheme to optimize')
            
class OneDimEmpWasserstein():
    def __init__(self, N):
        # define integration points
        w, q = compute_clenshaw_curtis(N)
        w, q = transform_points(w, q, a=0., b=1.)
        # remove CC end points
        self.w = w[1:-1,0]
        self.q = q[1:-1,0]
        
    def forward(self, nu, Tx, p):
        """ Compute Wasserstein-p distance given N integration points"""
        # evaluate empirical inverse CDFs
        inv_cdf1x = nu.icdf(self.q)
        inv_cdf2x = torch.quantile(Tx, self.q)
        # evaluate Ix and compute inner product
        Ix = torch.abs(inv_cdf1x - inv_cdf2x).pow(p)
        return torch.dot(self.w, Ix).pow(1./p)
