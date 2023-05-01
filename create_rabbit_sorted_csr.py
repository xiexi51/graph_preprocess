from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import local_setting

base_path = local_setting.base_path

fileset = Path(base_path).rglob('*.config')

n = 0

for file in fileset:
    print(file.stem)
    if file.stem[:6] == "rabbit":
        continue
    # if file.stem != "ppi":
    #     continue
    indptr = np.fromfile(base_path + "rabbit_" + file.stem + ".graph.ptrdump", dtype=np.int32)
    indices = np.fromfile(base_path + "rabbit_" + file.stem + ".graph.edgedump", dtype=np.int32)
    num_nodes = len(indptr) - 1
    num_edges = len(indices)
    vals = np.ones(num_edges)
    csr = csr_matrix((vals, indices, indptr))

    degrees = np.ediff1d(indptr)
    sorted_deg_arg = np.argsort(degrees, kind="stable")
    sorted_deg = degrees[sorted_deg_arg]

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

    new_indptr.astype(np.int32).tofile(base_path + "rabbit_" + file.stem + '.new_indptr')
    new_indices.astype(np.int32).tofile(base_path + "rabbit_" + file.stem + '.new_indices')