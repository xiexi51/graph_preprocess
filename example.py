import numpy as np
from scipy.sparse import csr_matrix
import torch

indptr = np.fromfile("cora.graph.ptrdump", dtype=np.int32)
indices = np.fromfile("cora.graph.edgedump", dtype=np.int32)
v_num = len(indptr) - 1
e_num = len(indices)
vals = np.ones(e_num)
A_csr = csr_matrix((vals, indices, indptr))

# randomly generate B, then randomly "top-k" selects it, that is enough for kernel speed testing
dim_B = 256
dim_sparse = 30

B = np.random.rand(v_num, dim_B)

sparsified_B_dense = np.zeros_like(B)

for i in range(B.shape[0]):
    indices = np.random.choice(B.shape[1], dim_sparse, replace=False)
    sparsified_B_dense[i, indices] = B[i, indices]

sparsified_B_csr = csr_matrix(sparsified_B_dense)

# SPMM
result_SPMM = A_csr.dot(sparsified_B_dense)

# SPGEMM
result_SPGEMM = A_csr.dot(sparsified_B_csr)

# check if the two results are close
result_SPGEMM_todense = result_SPGEMM.toarray()

if(np.allclose(result_SPGEMM_todense, result_SPMM)):
    print("yes")
else:
    print("no")

mask = torch.zeros_like(B, dtype=torch.float32, requires_grad=False).cuda()

with torch.no_grad():
    rank_b = torch.argsort(B, dim=-1, descending=True)
    rank_indice = torch.argsort(rank_b, dim=-1, descending=False)
    mask[rank_indice < dim_sparse] = 1


