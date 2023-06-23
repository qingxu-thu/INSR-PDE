import torch
print('PyTorch version:',torch.__version__)

torchdevice = torch.device('cpu')
if torch.cuda.is_available():
  torchdevice = torch.device('cuda')
  print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torchdevice))

import cupy as cp
import cupyx.scipy.sparse.linalg
print('CuPy version:',cp.__version__)
print('Running on ',cp.array([1]).data.device)

import numpy as np
import time
# Convenience function to map a torch COO tensor in a cupy one
def coo_torch2cupy(A):
  A = A.data.coalesce()
  Avals_cp = cp.asarray(A.values())
  Aidx_cp = cp.asarray(A.indices())
  return cp.sparse.coo_matrix((Avals_cp, Aidx_cp),shape=(A.shape[0],A.shape[1]))

# Custom PyTorch sparse solver exploiting a CuPy backend
# See https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
class SparseSolve(torch.autograd.Function):
  @staticmethod
  def forward(ctx, A, b):
    # Sanity check
    if A.ndim != 2 or (A.shape[0] != A.shape[1]):
      raise ValueError("A should be a square 2D matrix.")
    # Transfer data to CuPy
    A_cp = coo_torch2cupy(A)
    b_cp = cp.asarray(b.data)
    # Solver the sparse system
    ctx.factorisedsolver = None
    if (b.ndim == 1) or (b.shape[1] == 1):
      # cp.sparse.linalg.spsolve only works if b is a vector but is fully on GPU
      x_cp = cp.sparse.linalg.spsolve(A_cp, b_cp)
    else:
      # Make use of a factorisation (only the solver is then on the GPU)
      # We store it in ctx to reuse it in the backward pass
      ctx.factorisedsolver = cp.sparse.linalg.factorized(A_cp)
      x_cp = ctx.factorisedsolver(b_cp)
    # Transfer (dense) result back to PyTorch
    x = torch.as_tensor(x_cp, device=b.device)
    if A.requires_grad or b.requires_grad:
      # Not sure if the following is needed / helpful
      x.requires_grad = True
    else:
      # Free up memory
      ctx.factorisedsolver = None
    # Save context for backward pass
    ctx.save_for_backward(A, b, x)
    return x

  @staticmethod
  def backward(ctx, grad):
    # Recover context
    A, b, x = ctx.saved_tensors
    # Compute gradient with respect to b
    if (ctx.factorisedsolver is None):
      gradb = SparseSolve.apply(A.t(), grad)
    else:
      # Re-use factorised solver from forward pass
      grad_cp = cp.asarray(grad.data)
      gradb_cp = ctx.factorisedsolver(grad_cp, trans='T')
      gradb = torch.as_tensor(gradb_cp, device=b.device)
    # The gradient with respect to the (dense) matrix A would be something like
    # -gradb @ x.T but we are only interested in the gradient with respect to
    # the (non-zero) values of A
    gradAidx = A.indices()
    mgradbselect = -gradb.index_select(0,gradAidx[0,:])
    xselect = x.index_select(0,gradAidx[1,:])
    mgbx = mgradbselect * xselect
    if x.dim() == 1:
      gradAvals = mgbx
    else:
      gradAvals = torch.sum( mgbx, dim=1 )
    gradAs = torch.sparse_coo_tensor(gradAidx, gradAvals, A.shape)
    return gradAs, gradb

sparse_solve = SparseSolve.apply