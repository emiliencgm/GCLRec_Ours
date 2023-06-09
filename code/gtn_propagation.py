'''
Modified by Guanming Chen from GTN:
## GTN-pytorch
"Graph Trend Filtering Networks for Recommendations", Accepted by SIGIR'2022.
Pytorch Implementation of GTN in Graph Trend Filtering Networks for Recommendations
The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch

@inproceedings{fan2022graph,
  title={Graph Trend Filtering Networks for Recommendations},
  author={Fan, Wenqi and Liu, Xiaorui and Jin, Wei and Zhao, Xiangyu and Tang, Jiliang and Li, Qing},
  booktitle={International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2022}
}
'''

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing

from torch_sparse import sum, mul, fill_diag, remove_diag


class GeneralPropagation(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K, alpha, cached=False, add_self_loops=True, add_self_loops_l1=True, normalize=True, **kwargs):
        super(GeneralPropagation, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.add_self_loops_l1 = add_self_loops_l1
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

    def get_incident_matrix(self, edge_index: Adj):
        size = edge_index.sizes()[1]
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        mask = row_index >= col_index#矩阵右上角mask为False
        row_index = row_index[mask]
        col_index = col_index[mask]
        edge_num = row_index.numel()#mask后剩下的边的条数
        row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
        col = torch.cat([row_index, col_index])
        value = torch.cat([torch.ones(edge_num), -1 * torch.ones(edge_num)]).cuda()
        inc = SparseTensor(row=row, rowptr=None, col=col, value=value, sparse_sizes=(edge_num, size))
        return inc

    def inc_norm(self, inc, edge_index, add_self_loops):
        if add_self_loops:
            edge_index = fill_diag(edge_index, 1.0)
        else:
            edge_index = remove_diag(edge_index)
        deg = sum(edge_index, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        inc = mul(inc, deg_inv_sqrt.view(1, -1))  ## col-wise 右乘一个对角阵==每列乘对应元素
        return inc


    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')

            elif isinstance(edge_index, SparseTensor):
                ## first cache incident_matrix (before normalizing edge_index)
                cache = self._cached_inc
                if cache is None:
                    incident_matrix = self.get_incident_matrix(edge_index=edge_index)
                    incident_matrix = self.inc_norm(inc=incident_matrix, edge_index=edge_index, add_self_loops=self.add_self_loops_l1)
                
                    if self.cached:
                        self._cached_inc = incident_matrix
                        self.init_z = torch.zeros((incident_matrix.sizes()[0], x.size()[-1])).cuda()# num_edge(interact) * dim_embed
                else:
                    incident_matrix = self._cached_inc

        K_ = self.K
        assert edge_weight is None
        if K_ <= 0:
            return x

        hh = x

        x, xs = self.gtn_forward(x=x, hh=hh, incident_matrix=incident_matrix, K=K_)
        return x, xs

    def gtn_forward(self, x, hh, K, incident_matrix):
        lambda2 = 4.0 #min(|Y-^(k+1)|,lambda2)
        beta = 0.5
        gamma = 1

        z = self.init_z.detach()

        xs = []
        for k in range(K):
            grad = x - hh
            smoo = x - gamma * grad
            temp = z + beta / gamma * (incident_matrix @ (smoo - gamma * (incident_matrix.t() @ z)))#(smoo - gamma * (incident_matrix.t() @ z)) == E-^(k+1) 式(10)

            z = self.proximal_l1_conjugate(x=temp, lambda2=lambda2, beta=beta, gamma=gamma, m="L1")

            ctz = incident_matrix.t() @ z

            x = smoo - gamma * ctz

        light_out = x

        return light_out, xs

    def proximal_l1_conjugate(self, x: Tensor, lambda2, beta, gamma, m):
        if m == 'L1':
            x_pre = x
            x = torch.clamp(x, min=-lambda2, max=lambda2)
            # print('diff after proximal: ', (x-x_pre).norm())

        elif m == 'L1_original':  ## through conjugate
            rr = gamma / beta
            yy = rr * x
            x_pre = x
            temp = torch.sign(yy) * torch.clamp(torch.abs(yy) - rr * lambda2, min=0)
            x = x - temp / rr

        else:
            raise ValueError('wrong prox')
        return x

    # def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
    #     return edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
    #     return matmul(adj_t, x, reduce=self.aggr)

    # def __repr__(self):
    #     return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K, self.alpha)