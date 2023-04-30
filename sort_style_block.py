from pathlib import Path
from itertools import chain
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import local_setting

fileset = Path(local_setting.base_path).glob('*.new_indptr')

for file in fileset:
    if file.stem == 'cora_modify':
        continue
    # if not (file.stem == 'cora' or file.stem == 'youtube' or file.stem == 'artist' or file.stem == 'pubmed' or file.stem == 'reddit.dgl' or file.stem == 'ppa' or file.stem == 'protein'):
    #     continue
    print(file.stem)

    new_indptr = np.fromfile(local_setting.base_path + file.stem + ".new_indptr", dtype=np.int32)
    new_indices = np.fromfile(local_setting.base_path + file.stem + ".new_indices", dtype=np.int32)
    v_num = len(new_indptr) - 1
    e_num = len(new_indices)
    vals = np.ones(e_num)
    new_csr = csr_matrix((vals, new_indices, new_indptr))
    print(v_num, e_num)

    degrees = np.ediff1d(new_csr.indptr)
    sorted_deg_arg = np.argsort(degrees)
    sorted_deg = degrees[sorted_deg_arg]

    d0_start = (degrees == 0).sum()

    cur_row = 0
    cur_loc = 0
    cur_degree = 0
    block_degree = []
    block_row_begin = []
    block_loc_begin = []

    block_info = []

    deg_bound = 12 * 32
    num_warps = 12

    warp_nz = [0]
    d_block_rows = [0]
    warp_max_nz = deg_bound // num_warps
    factor = [1, 2, 3, 4, 6, 12]
    jf = 0
    i = 1
    while i < deg_bound // 2:
        if factor[jf] * warp_max_nz >= i:
            warp_nz.append((i + factor[jf] - 1) // factor[jf])
            d_block_rows.append(num_warps // factor[jf])
            i += 1
        else:
            jf += 1

    while True:
        if sorted_deg[cur_row] != cur_degree:
            cur_degree = sorted_deg[cur_row]

        if cur_degree == 0:
            cur_row += 1
            cur_loc += 1

        elif cur_degree >= 1 and cur_degree <= deg_bound:
            if cur_degree >= len(warp_nz):
                w_nz = deg_bound // num_warps
            else:
                w_nz = warp_nz[cur_degree]
            if cur_degree >= len(d_block_rows):
                b_row = 1
            else:
                b_row = d_block_rows[cur_degree]

            block_row_begin.append(cur_row)
            block_loc_begin.append(cur_loc)

            j = 0
            while sorted_deg[cur_row] == cur_degree:
                cur_row += 1
                j += 1
                if j == b_row:
                    break
                if cur_row == len(new_indptr) - 1:
                    break
            cur_loc += j * cur_degree
            block_degree.append(cur_degree)
            block_info.append((w_nz << 16) + j)

        elif cur_degree > deg_bound:
            tmp_loc = 0
            while True:
                block_degree.append(cur_degree)
                block_row_begin.append(cur_row)
                block_loc_begin.append(cur_loc)
                if tmp_loc + deg_bound > cur_degree:
                    block_info.append(cur_degree - tmp_loc)
                    cur_loc += cur_degree - tmp_loc
                    tmp_loc = cur_degree
                else:
                    block_info.append(deg_bound)
                    tmp_loc += deg_bound
                    cur_loc += deg_bound
                if tmp_loc == cur_degree:
                    break
            cur_row += 1

        else:
            print("cur_degree number is wrong")
            break

        if cur_row == len(new_indptr) - 1:
            print(cur_row)
            break

    block_4 = np.dstack([block_degree, block_row_begin, block_loc_begin, block_info]).flatten()
    # block_4.astype(np.int32).tofile('./graphs/' + file.stem + '.block4')
    block_4.astype(np.int32).tofile('./block_4/' + file.stem + '.block4')