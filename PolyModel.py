import torch
import torch.nn as nn
import numpy as np
import math
from scipy.special import gamma

PI = torch.tensor(math.pi)

class LinearExpansion(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.linear = torch.nn.Linear(degree+1, 1, bias=False)

    def forward(self, x):
        xp = self.eval_basis(x).detach()
        return self.linear(xp)

    def grad_x(self, x):
        xp = self.grad_x_basis(x).detach()
        return self.linear(xp)

class HermitePolyModel(LinearExpansion):
    def __init__(self, degree):
        super(HermitePolyModel, self).__init__(degree)

    def eval_basis(self, x):
        assert(x.ndim == 1)
        H = torch.zeros(x.size(0), self.degree+1).type(x.type())
        H[:, 0] = x * 0 + 1.
        if self.degree > 0:
            H[:, 1] = x
            # apply recursion for polynomials
            for ii in range(1, self.degree):
                H[:, ii+1] = x * H[:, ii] - ii * H[:, ii-1]
        return H

    def grad_x_basis(self, x):
        assert(x.ndim == 1)
        dxH = torch.zeros(x.size(0), self.degree+1).type(x.type())
        dxH[:, 0] = torch.zeros(x.size(0))
        if self.degree > 0:
            # evaluate polynomials
            H = self.eval_basis(x)
            dxH[:, 1:] = H[:,:-1] * torch.arange(1,self.degree+1)[:,None].T
        return dxH

    def normalization_const(self):
        norm_const = np.sqrt(np.sqrt(2*np.pi)*gamma(np.arange(0,self.degree+1)+1))
        return torch.tensor(norm_const)


class HermiteFunctionModel(LinearExpansion):
    def __init__(self, degree):
        super(HermiteFunctionModel, self).__init__(degree)

    def eval_basis(self, x):
        assert(x.ndim == 1)
        H = torch.zeros(x.size(0), self.degree+1).type(x.type())
        H[:, 0] = torch.exp(-x.pow(2)/2.) / (PI**(1/4.))
        if self.degree > 0:
            H[:, 1] = x * torch.exp(-x.pow(2)/2.) * torch.sqrt(torch.tensor(2)) / (PI**(1/4.))
            # apply recursion for functions
            for ii in range(1, self.degree):
                H[:, ii+1] = x * H[:, ii] - torch.sqrt(torch.tensor(ii/2.)) * H[:, ii-1]
                H[:, ii+1] /= torch.sqrt( torch.tensor((ii+1.)/2.) )
        return H

    def grad_x_basis(self, x):
        assert(x.ndim == 1)
        # declare arrays for both evaluations and derivatives
        H   = torch.zeros(x.size(0), self.degree+1).type(x.type())
        dxH = torch.zeros(x.size(0), self.degree+1).type(x.type())
        # initialize
        H[:, 0]   = torch.exp(-x.pow(2)/2.) / (PI**(1/4.))
        dxH[:, 0] = H[:, 0]*(-1.*x)
        if self.degree > 0:
            H[:, 1]   = x * torch.exp(-x.pow(2)/2.) * torch.sqrt(torch.tensor(2)) / (PI**(1/4.))
            dxH[:, 1] = (1 - x.pow(2)) * torch.exp(-x.pow(2)/2.) * torch.sqrt(torch.tensor(2)) / (PI**(1/4.))
            # apply recursion for derivatives
            for ii in range(1, self.degree):
                H[:, ii+1]   = x * H[:, ii] - torch.sqrt(torch.tensor(ii/2.)) * H[:, ii-1]
                H[:, ii+1]   /= torch.sqrt( torch.tensor((ii+1.)/2.) )
                dxH[:, ii+1] = torch.sqrt(torch.tensor(2.*(ii+1))) * H[:, ii] - x * H[:, ii+1]
        return dxH 


class LegendrePolyModel(LinearExpansion):
    def __init__(self, degree):
        super(LegendrePolyModel, self).__init__(degree)

    def eval_basis(self, x):
        assert(x.ndim == 1)
        H = torch.zeros(x.size(0), self.degree+1).type(x.type())
        # initialize
        H[:, 0] = x * 0 + 1.
        if self.degree > 0:
            H[:, 1] = x
            # apply recursion for polynomials
            for ii in range(1, self.degree):
                H[:, ii+1] = ( (2*ii+1)*x*H[:, ii] - ii*H[:, ii-1] ) / (ii+1)
        return H

    def grad_x_basis(self, x):
        assert(x.ndim == 1)
        # declare arrays for both evaluations and derivatives
        H   = torch.zeros(x.size(0), self.degree+1).type(x.type())
        dxH = torch.zeros(x.size(0), self.degree+1).type(x.type())
        # initialize
        H[:, 0] = x * 0 + 1.
        if self.degree > 0:
            H[:, 1]   = x
            dxH[:, 1] = torch.ones(x.size(0))
            # apply recursion for derivatives
            for ii in range(1, self.degree):
                H[:, ii+1]   = ( (2*ii+1)*x*H[:, ii] - ii*H[:, ii-1] ) / (ii+1)
                dxH[:, ii+1] = ( (2*ii+1)*(H[:, ii] + x*dxH[:, ii]) - ii*dxH[:, ii-1] ) / (ii+1)
        return dxH

    def normalization_const(self):
        norm_const = np.sqrt(2./(2*np.arange(0,self.degree+1)+1))
        return torch.tensor(norm_const)


