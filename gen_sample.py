import numpy as np
import random
from scipy import sparse
from scipy.sparse import csr_matrix

random.seed(1234)

row = 40
col = 40
A = np.zeros((row,col))
n = 1001
for i in range(0,row):
    rnz = random.randint(0,5)
    if i == 18:
        rnz = 23
    location = random.sample(list(range(0,col)), rnz)
    for j in sorted(location):
        A[i, j] = n
        n += 1

S = sparse.csr_matrix(A)
# sparse.save_npz("npz1.npz", S)
print(S.nnz)

degrees = np.ediff1d(S.indptr)
sorted_deg_arg = np.argsort(degrees)
print(degrees[sorted_deg_arg])

sorted_deg = degrees[sorted_deg_arg]

nz = 0
new_indptr = np.zeros_like(S.indptr)
new_indices = np.zeros_like(S.indices)
new_data = np.zeros_like(S.data)
new_indptr[0] = nz

for i in range(len(degrees)):
    new_r_begin = S.indptr[sorted_deg_arg[i]]
    new_r_end = S.indptr[sorted_deg_arg[i] + 1]
    new_indices[nz : nz + sorted_deg[i]] = S.indices[new_r_begin : new_r_end]
    new_data[nz: nz + sorted_deg[i]] = S.data[new_r_begin: new_r_end]
    nz += sorted_deg[i]
    new_indptr[i + 1] = nz

new_csr = csr_matrix((new_data, new_indices, new_indptr))

new_A = new_csr.todense()

for i in range(row):
    for j in range(col):
        print(int(A[i, j]), end=" ")
    print("")
print("\n")

for i in range(row):
    for j in range(col):
        print(int(new_A[i, j]), end=" ")
    print("")