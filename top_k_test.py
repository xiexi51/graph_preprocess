import numpy as np
from scipy.sparse import csr_matrix
import torch
import local_setting
import time


graph = "reddit.dgl"
# indptr = np.fromfile(local_setting.base_path + graph + ".graph.ptrdump", dtype=np.int32)
# indices = np.fromfile(local_setting.base_path + graph + ".graph.edgedump", dtype=np.int32)
# v_num = len(indptr) - 1
# e_num = len(indices)
# vals = np.ones(e_num)
# A_csr = csr_matrix((vals, indices, indptr))

v_num = 232965

# randomly generate B, then randomly "top-k" selects it, that is enough for kernel speed testing
dim_B = 512
dim_sparse = 100


torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory
# less than 0.5 will be ok:
tmp_tensor = torch.empty(int(v_num * dim_B), dtype=torch.int8, device='cuda')
del tmp_tensor
torch.cuda.empty_cache()


with torch.no_grad():
    B_tensor = torch.rand((v_num, dim_B), device='cuda')

    topk_vals = torch.zeros((v_num, dim_sparse), dtype=torch.float32, device='cuda')
    topk_indices = torch.zeros((v_num, dim_sparse), dtype=torch.int32, device='cuda')

    torch.cuda.synchronize()
    for _ in range(20):
        B_tensor.topk(dim_sparse, dim=-1, sorted=False)
    
    torch.cuda.synchronize()
    start_prop = time.perf_counter()

    for _ in range(200):
        B_tensor.topk(dim_sparse, dim=-1, sorted=False)
        torch.cuda.synchronize()

    t = time.perf_counter() - start_prop

    print(t / 200)



# print(topk_vals.shape, topk_indices.shape)
