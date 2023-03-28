import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

file = "/home/xiexi/py_projects/OSDI21_AE/osdi-ae-graphs/cora.npz"

graph_obj = np.load(file)
src_li = graph_obj['src_li']
dst_li = graph_obj['dst_li']
num_nodes = graph_obj['num_nodes']
num_edges = len(src_li)
edge_index = np.stack([src_li, dst_li])
val = [1] * num_edges
scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
dense = scipy_coo.todense()
dense[40:70, :] = 0

coo = coo_matrix(dense)
# scipy_csr = scipy_coo.tocsr()
print(coo.shape)
csr = coo.tocsr()

degrees = np.ediff1d(csr.indptr)
sorted_deg_arg = np.argsort(degrees)
sorted_deg = degrees[sorted_deg_arg]

print("degrees==0 ", (degrees == 0).sum() / num_nodes)
print("degrees==1 ", (degrees == 1).sum() / num_nodes)
print("degrees<=2 ", (degrees <= 2).sum() / num_nodes)
print("degrees<=4 ", (degrees <= 4).sum() / num_nodes)

# print(degrees[sorted_deg_arg])

nz = 0
new_indptr = np.zeros_like(csr.indptr)
new_indices = np.zeros_like(csr.indices)
new_data = np.zeros_like(csr.data)
new_indptr[0] = nz

for i in range(len(degrees)):
    new_r_begin = csr.indptr[sorted_deg_arg[i]]
    new_r_end = csr.indptr[sorted_deg_arg[i] + 1]
    new_indices[nz: nz + sorted_deg[i]] = csr.indices[new_r_begin: new_r_end]
    new_data[nz: nz + sorted_deg[i]] = csr.data[new_r_begin: new_r_end]
    nz += sorted_deg[i]
    new_indptr[i + 1] = nz

new_csr = csr_matrix((new_data, new_indices, new_indptr))

print(new_indptr.shape)
print(new_indices.shape)

new_indptr.astype(np.int32).tofile('./graphs/' + 'cora_modify' + '.new_indptr')
new_indices.astype(np.int32).tofile('./graphs/' + 'cora_modify' + '.new_indices')