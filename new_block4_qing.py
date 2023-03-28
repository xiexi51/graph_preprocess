from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

fileset = Path('qing_data').rglob('*.config')

for file in fileset:
    print(file.stem)
    new_indptr = np.fromfile('qing_data/' + file.stem + ".new_indptr", dtype=np.int32)
    new_indices = np.fromfile('qing_data/' + file.stem + ".new_indices", dtype=np.int32)
    v_num = len(new_indptr) - 1
    e_num = len(new_indices)
    vals = np.ones(e_num)
    new_csr = csr_matrix((vals, new_indices, new_indptr))
    print(v_num, e_num)

    degrees = np.ediff1d(new_csr.indptr)
    sorted_deg_arg = np.argsort(degrees)
    sorted_deg = degrees[sorted_deg_arg]

    d0_start = (degrees == 0).sum()
    print("degrees==0 ", d0_start / v_num)
    print("degrees==1 ", (degrees == 1).sum() / v_num)
    print("degrees<=2 ", (degrees <= 2).sum() / v_num)
    print("degrees<=4 ", (degrees <= 4).sum() / v_num)


    cur_row = 0
    cur_loc = 0
    cur_degree = 0
    block_degree = []
    block_row_begin = []
    block_loc_begin = []
    # block_loc_end = []
    block_info = []
    # #deg                1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96
    # #warps             12  12  12  12  12  12  12  12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12  9  9  9 10 10 10 11 11 11 12 12 12 10
    # warp_nz   =    [0,  1,  2,  3,  4,  5,  6,  7,  8,  5,  5,  6,  6,  7,  7,  8,  8, 6, 6, 7, 7, 7, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    # d_block_rows = [0, 24, 24, 24, 24, 24, 24, 24, 24, 12, 12, 12, 12, 12, 12, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # deg_bound = 192


    # #deg                1   2   3   4   5   6   7   8   9  10  11  12 13 14 15 16 17 18  19  20  21  22  23  24 25 26 27 28 29 30 31 32 33 34 35 36  37  38  39  40  41  42  43  44  45  46  47  48 49 50 51 52 53 54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72 73 74 75 76 77 78
    # #warps             12  12  12  12  12  12  12  12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12  9  9  9 10 10 10 11 11 11 12 12 12 10
    # warp_nz   =    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12]
    # d_block_rows = [0, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 6, 6, 4, 4,  4,  4,  4,  4,  4,  4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 2, 2, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2]
    # deg_bound = 144

    # #deg                1   2   3   4   5   6   7   8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61
    # #warps             12  12  12  12  12  12  12  12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12  9  9  9 10 10 10 11 11 11 12 12 12 10
    # warp_nz   =    [0,  1,  2,  3,  4,  5,  6,  7,  8, 5, 5, 6, 6, 7, 7, 8, 8, 6, 6, 7, 7, 7, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8]
    # d_block_rows = [0, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # deg_bound = 96

    # #deg                1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16 17 18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96
    # #warps             12  12  12  12  12  12  12  12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12  9  9  9 10 10 10 11 11 11 12 12 12 10
    # warp_nz   =    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16]
    # d_block_rows = [0, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2]

    deg_bound = 384
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
    block_4.astype(np.int32).tofile('./qing_data/' + file.stem + '.block4')

