import cupy as cp

def coo_torch2cupy(A):
  A = A.data.coalesce()
  Avals_cp = cp.asarray(A.values())
  Aidx_cp = cp.asarray(A.indices())
  return cp.sparse.coo_matrix((Avals_cp, Aidx_cp))