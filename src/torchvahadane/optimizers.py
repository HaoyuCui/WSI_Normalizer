"""
code adapted from https://github.com/rfeinman/pytorch-lasso
"""
import math
from scipy.sparse.linalg import eigsh
import torch
from typing import Union
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import warnings

def ridge(b, A, alpha=1e-4):
    # right-hand side
    rhs = torch.matmul(A.T, b)
    # regularized gram matrix
    M = torch.matmul(A.T, A)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    if info != 0:
        raise RuntimeError("The Gram matrix is not positive definite. "
                           "Try increasing 'alpha'.")
    x = torch.cholesky_solve(rhs, L)
    return x

def initialize_code(x, weight, alpha, mode):
    n_samples = x.size(0)
    n_components = weight.size(1)
    if mode == 'zero':
        z0 = x.new_zeros(n_samples, n_components)
    elif mode == 'unif':
        z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1)
    elif mode == 'transpose':
        z0 = torch.matmul(x, weight)
    elif mode == 'ridge':
        z0 = ridge(x.T, weight, alpha=alpha).T
    else:
        raise ValueError("invalid init parameter '{}'.".format(mode))

    return z0


def coord_descent(x, W, z0=None, alpha=1.0, lambda1=0.01, maxiter=1000, tol=1e-6, verbose=False, positive=True):
    """ modified coord_descent"""
    input_dim, code_dim = W.shape  # [D,K]
    batch_size, input_dim1 = x.shape  # [N,D]
    assert input_dim1 == input_dim
    tol = tol * code_dim
    if z0 is None:
        z = x.new_zeros(batch_size, code_dim)  # [N,K]
    else:
        assert z0.shape == (batch_size, code_dim)
        z = z0

    # initialize b
    # TODO: how should we initialize b when 'z0' is provided?
    b = torch.mm(x, W)  # [N,K]

    # precompute S = I - W^T @ W
    S = - torch.mm(W.T, W)  # [K,K]
    S.diagonal().add_(1.)

    def fn(z):
        x_hat = torch.matmul( W, z.T)
        loss = 0.5 * (x- x_hat).norm(p=2).pow(2) + z.norm(p=1)*lambda1
        return loss

    def cd_update(z, b):
        if positive:
            z_next  = torch.clamp(b.abs() - alpha, min=0)
        else:
            z_next = F.softshrink(b, alpha)  # [N,K]
        z_diff = z_next - z  # [N,K]
        k = z_diff.abs().argmax(1)  # [N]
        kk = k.unsqueeze(1)  # [N,1]
        b = b + S[:,k].T * z_diff.gather(1, kk)  # [N,K] += [N,K] * [N,1]
        z = z.scatter(1, kk, z_next.gather(1, kk))
        return z, b

    active = torch.arange(batch_size, device=W.device)
    for i in range(maxiter):
        if len(active) == 0:
            break
        z_old = z[active]
        z_new, b[active] = cd_update(z_old, b[active])
        update = (z_new - z_old).abs().sum(1)
        z[active] = z_new
        active = active[update > tol]
        if verbose:
            print('iter %i - loss: %0.4f' % (i, fn(F.softshrink(b, alpha))))

    if positive:
        z  = torch.clamp(b.abs() - alpha, min=0)
    else:
        z = F.softshrink(b, alpha)  # [N,K]

    return z


def _lipschitz_constant(W):
    #L = torch.linalg.norm(W, ord=2) ** 2
    WtW = torch.matmul(W.t(), W)
    # L = torch.linalg.eigvalsh(WtW)[-1]
    L = eigsh(WtW.detach().cpu().numpy(), k=1, which='LM',
              return_eigenvectors=False).item()

    if not np.isfinite(L):  # sometimes L is not finite because of potential cublas error.
        L = torch.linalg.norm(W, ord=2) ** 2
    return L


def ista(x, z0, weight, alpha=1.0, fast=True, lr='auto', maxiter=250,
         tol=1e-10, backtrack=False, eta_backtrack=1.5, verbose=False, positive=True, cut=200):

    if type(z0) is str:
        z0 = initialize_code(x, weight, alpha, z0)

    if lr == 'auto':
        # set lr based on the maximum eigenvalue of W^T @ W; i.e. the
        # Lipschitz constant of \grad f(z), where f(z) = ||Wz - x||^2
        L = _lipschitz_constant(weight)
        lr = 1 / L
    tol = z0.numel() * tol

    def loss_fn(z_k):
        x_hat = torch.matmul(weight, z_k.T)
        loss = 0.5 * (x.T-x_hat).norm(p=2).pow(2) + z_k.norm(p=1)*alpha
        return loss

    def rss_grad(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        return torch.matmul(resid, weight)

    # optimize
    z = z0
    if fast:
        y, t = z0, 1
    for i in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z))
        # ista update
        z_prev = y if fast else z

        if positive:
            # IPTA update rule based on https://mayhhu.github.io/pdf/2018_L1-NNSO-Optim_ZHYW.pdf
            z_next = torch.clamp((torch.abs(z_prev - lr * rss_grad(z_prev))-(alpha*lr)), min=0)  # positive shrinkage operation
        else:
            z_next = F.softshrink(z_prev - lr * rss_grad(z_prev), alpha * lr)
        # check convergence
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            # print('reached')
            break

        # update variables
        if fast and i<cut:
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        z = z_next
    return z
