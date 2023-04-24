from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import torch
import rabbit
import os.path

base_path = '/home/xi/cuda_projects/hpc_data/'

fileset = Path(base_path).rglob('*.config')

n = 0

for file in fileset:
    print(file.stem)
    
    if file.stem != 'artist':
        continue
    # if os.path.exists(base_path + 'rabbit_' + file.stem + '.config'):
    #     print("rabbit exists")
    #     continue

    indptr = np.fromfile(base_path + file.stem + ".graph.ptrdump", dtype=np.int32)
    indices = np.fromfile(base_path + file.stem + ".graph.edgedump", dtype=np.int32)
    num_nodes = len(indptr) - 1
    num_edges = len(indices)
    vals = np.ones(num_edges)
    csr = csr_matrix((vals, indices, indptr))
    coo = csr.tocoo()
    
    edge_index = np.stack([coo.row, coo.col])
    val = [1] * num_edges

    rabbit_edge_index = rabbit.reorder(torch.IntTensor(edge_index))
    
    rabbit_coo = coo_matrix((val, rabbit_edge_index), shape=(num_nodes, num_nodes))
    rabbit_csr = rabbit_coo.tocsr()
    
    with open(base_path + 'rabbit2_' + file.stem + '.config', 'w') as conf:
        conf.write(str(num_nodes) + " " + str(num_edges))
    rabbit_csr.indptr.astype(np.int32).tofile(base_path + 'rabbit2_' + file.stem + '.graph.ptrdump')
    rabbit_csr.indices.astype(np.int32).tofile(base_path + 'rabbit2_' + file.stem + '.graph.edgedump')


