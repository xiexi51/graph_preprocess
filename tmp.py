from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

fileset = Path('qing_data').rglob('*.config')

for file in fileset:
    print(file.stem)
    indptr = np.fromfile('qing_data/' + file.stem + ".graph.ptrdump", dtype=np.int32)
    indices = np.fromfile('qing_data/' + file.stem + ".graph.edgedump", dtype=np.int32)
    v_num = len(indptr) - 1
    e_num = len(indices)
    vals = np.ones(e_num)
    csr = csr_matrix((vals, indices, indptr))
    print(v_num, e_num)
    degrees = np.ediff1d(csr.indptr)
    sorted_deg_arg = np.argsort(degrees)
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

    new_indptr.astype(np.int32).tofile('qing_data/' + file.stem + '.new_indptr')
    new_indices.astype(np.int32).tofile('qing_data/' + file.stem + '.new_indices')

    # new_csr = csr_matrix((new_data, new_indices, new_indptr))

    d0_start = (degrees == 0).sum()
    print("degrees==0 ", d0_start / v_num)
    print("degrees==1 ", (degrees == 1).sum() / v_num)
    print("degrees<=2 ", (degrees <= 2).sum() / v_num)
    print("degrees<=4 ", (degrees <= 4).sum() / v_num)

