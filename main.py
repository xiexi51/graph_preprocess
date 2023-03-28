from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import torch
import rabbit
import matplotlib.pyplot as plt

fileset = Path('/home/xiexi/py_projects/OSDI21_AE/osdi-ae-graphs').rglob('*.npz')

n = 0

for file in fileset:
    print(file.stem)
    if file.stem != 'artist':
        continue
    # if i > 2:
    #     break

    graph_obj = np.load(file)
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_nodes = graph_obj['num_nodes']
    num_edges = len(src_li)
    edge_index = np.stack([src_li, dst_li])
    val = [1] * num_edges
    scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
    scipy_csr = scipy_coo.tocsr()

    degrees = np.ediff1d(scipy_csr.indptr)
    sorted_deg_arg = np.argsort(degrees)
    sorted_deg = degrees[sorted_deg_arg]

    print("degrees==1 ", (degrees == 1).sum() / num_nodes)
    print("degrees<=2 ", (degrees <= 2).sum() / num_nodes)
    print("degrees<=4 ", (degrees <= 4).sum() / num_nodes)

    # print(degrees[sorted_deg_arg])

    nz = 0
    new_indptr = np.zeros_like(scipy_csr.indptr)
    new_indices = np.zeros_like(scipy_csr.indices)
    new_data = np.zeros_like(scipy_csr.data)
    new_indptr[0] = nz

    for i in range(len(degrees)):
        new_r_begin = scipy_csr.indptr[sorted_deg_arg[i]]
        new_r_end = scipy_csr.indptr[sorted_deg_arg[i] + 1]
        new_indices[nz: nz + sorted_deg[i]] = scipy_csr.indices[new_r_begin: new_r_end]
        new_data[nz: nz + sorted_deg[i]] = scipy_csr.data[new_r_begin: new_r_end]
        nz += sorted_deg[i]
        new_indptr[i + 1] = nz

    new_csr = csr_matrix((new_data, new_indices, new_indptr))

    print(new_indptr.shape)
    print(new_indices.shape)

    # dim = 32
    # vin = np.zeros([len(new_indptr) - 1, dim])
    # k = 0
    # for i in range(len(new_indptr) - 1):
    #     for j in range(dim):
    #         vin[i,j] = 0.01 * k
    #         k += 1
    #
    # res = new_csr * vin

    new_indptr.astype(np.int32).tofile('./graphs/' + file.stem + '.new_indptr')
    new_indices.astype(np.int32).tofile('./graphs/' + file.stem + '.new_indices')

    cur_row = 0
    cur_loc = 0
    cur_degree = 1
    block_degree = []
    block_row_begin = []
    block_loc_begin = []
    block_loc_end = []
    degrees_len = [8, 4, 2, 2]
    while True:
        if cur_degree >= 1 and cur_degree <= 4:
            block_degree.append(cur_degree)
            block_row_begin.append(cur_row)
            block_loc_begin.append(cur_loc)
            j = 0
            while sorted_deg[cur_row] == cur_degree:
                cur_row += 1
                j += 1
                if j == degrees_len[cur_degree - 1]:
                    break
            cur_loc += j * cur_degree
            block_loc_end.append(cur_loc)
        # elif cur_degree >= 5 and cur_degree <= 8:
        #     block_degree.append(cur_degree)
        #     block_row_begin.append(cur_row)
        #     block_loc_begin.append(cur_loc)
        #     cur_row += 1
        #     cur_loc += cur_degree
        #     block_loc_end.append(cur_loc)
        # elif cur_degree > 8:
        elif cur_degree >= 5:
            tmp_loc = 0
            while True:
                block_degree.append(cur_degree)
                block_row_begin.append(cur_row)
                block_loc_begin.append(cur_loc)
                if tmp_loc + 8 > cur_degree:
                    cur_loc += cur_degree - tmp_loc
                    tmp_loc = cur_degree
                else:
                    tmp_loc += 8
                    cur_loc += 8
                block_loc_end.append(cur_loc)
                if tmp_loc == cur_degree:
                    break
            cur_row += 1

        else:
            print("cur_degree number is wrong")
            break

        if cur_row == len(new_indptr) - 1:
            print(cur_row)
            break
        if sorted_deg[cur_row] != cur_degree:
            cur_degree = sorted_deg[cur_row]

    block_4 = np.dstack([block_degree, block_row_begin, block_loc_begin, block_loc_end]).flatten()
    block_4.astype(np.int32).tofile('./graphs/' + file.stem + '.block4')

    # block_degree = np.array(block_degree)
    # block_row_begin = np.array(block_row_begin)
    # block_loc_begin = np.array(block_loc_begin)
    # block_loc_end = np.array(block_loc_end)
    #
    # new_coo = new_csr.tocoo()

    print("here")

    # for i in range(sorted_deg):


    # with open('./graphs/' + file.stem + '.config','w') as conf:
    #     conf.write(str(num_nodes) + " " + str(num_edges))
    # scipy_csr.indptr.astype(np.int32).tofile('./graphs/' + file.stem + '.graph.ptrdump')
    # scipy_csr.indices.astype(np.int32).tofile('./graphs/' + file.stem + '.graph.edgedump')

    # fig = plt.figure(figsize = (10, 5))
    # ax1 = plt.subplot(1, 2, 1)
    # ax1.scatter(scipy_coo.row, scipy_coo.col, s=1)

    # rabbit_edge_index = rabbit.reorder(torch.IntTensor(edge_index))
    #
    # rabbit_coo = coo_matrix((val, rabbit_edge_index), shape=(num_nodes, num_nodes))
    # rabbit_csr = rabbit_coo.tocsr()
    #
    # # ax2 = plt.subplot(1, 2, 2)
    # # ax2.scatter(rabbit_coo.row, rabbit_coo.col, s=1)
    #
    # with open('./graphs/rabbit_' + file.stem + '.config', 'w') as conf:
    #     conf.write(str(num_nodes) + " " + str(num_edges))
    # rabbit_csr.indptr.astype(np.int32).tofile('./graphs/rabbit_' + file.stem + '.graph.ptrdump')
    # rabbit_csr.indices.astype(np.int32).tofile('./graphs/rabbit_' + file.stem + '.graph.edgedump')


