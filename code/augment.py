"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
from model import GNN_Encoder_edge_index
import numpy as np
from utils import randint_choice
import scipy.sparse as sp
import world
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from precalcul import precalculate
import time
from k_means import kmeans
import faiss
import torch_sparse
from scipy.sparse import csr_matrix
from dataloader import dataset
import torch_geometric


class Homophily:
    def __init__(self, model:LightGCN):
        self.model = model
        
    def get_homophily_batch(self, batch_user:torch.Tensor, batch_item:torch.Tensor, mode='not_in_batch'):
        '''
        return prob distribution of users and items in batch.
        '''
        with torch.no_grad():
            sigma = world.config['sigma_gausse']
            ncluster = world.config['n_cluster']
            #edge_index = self.model.dataset.Graph.cpu().indices()
            if mode == 'in_batch':
                embs_KMeans = torch.cat((self.model.embedding_user(batch_user), self.model.embedding_item(batch_item)), dim=0)
            else:
                embs_KMeans = torch.cat((self.model.embedding_user.weight, self.model.embedding_item.weight), dim=0)
            
            if ncluster > 64:
                embs_KMeans_numpy = embs_KMeans.detach().cpu().numpy()
                kmeans_faiss = faiss.Kmeans(world.config['latent_dim_rec'], ncluster, gpu=True)
                kmeans_faiss.train(embs_KMeans_numpy)
                centroids = torch.tensor(kmeans_faiss.centroids).to(world.device)
            else:
                cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
                centroids = cluster_centers.to(world.device)            
            
            logits = []
            embs_batch = torch.cat((self.model.embedding_user(batch_user), self.model.embedding_item(batch_item)), dim=0)
            for c in centroids:
                logits.append((-torch.square(embs_batch - c).sum(1)/sigma).view(-1, 1))
            logits = torch.cat(logits, axis=1)
            probs = F.softmax(logits, dim=1)
            #probs = F.normalize(logits, dim=1)# TODO
            #loss = F.l1_loss(probs[edge_index[0]], probs[edge_index[1]])
            batch_user_prob, batch_item_prob = torch.split(probs, [batch_user.shape[0], batch_item.shape[0]])
        return batch_user_prob, batch_item_prob

    def get_homophily_batch_any(self, batch_embs1:torch.Tensor, batch_embs2:torch.Tensor):
        '''
        return prob distribution of users and items in batch.
        '''
        sigma = world.config['sigma_gausse']
        ncluster = world.config['n_cluster']
        #edge_index = self.model.dataset.Graph.cpu().indices()
        embs_KMeans = torch.cat((batch_embs1, batch_embs2), dim=0)
        
        if ncluster > 99:
            embs_KMeans_numpy = embs_KMeans.detach().cpu().numpy()
            kmeans_faiss = faiss.Kmeans(world.config['latent_dim_rec'], ncluster, gpu=True)
            kmeans_faiss.train(embs_KMeans_numpy)
            centroids = torch.tensor(kmeans_faiss.centroids).to(world.device)
        else:
            cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
            centroids = cluster_centers.to(world.device)            
        
        logits = []
        for c in centroids:
            logits.append((-torch.square(embs_KMeans - c).sum(1)/sigma).view(-1, 1))
        logits = torch.cat(logits, axis=1)
        probs = F.softmax(logits, dim=1)
        #probs = F.normalize(logits, dim=1)# TODO
        #loss = F.l1_loss(probs[edge_index[0]], probs[edge_index[1]])
        batch_prob1, batch_prob2 = torch.split(probs, [batch_embs1.shape[0], batch_embs2.shape[0]])
        
        return batch_prob1, batch_prob2


class ED_Uniform():
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.augAdjMatrix1 = None
        self.augAdjMatrix2 = None

    def Edge_drop_random(self, p_drop):
        '''
        return: dropout后保留的交互构成的按度归一化的邻接矩阵(sparse)
        '''
        n_nodes = self.model.num_users + self.model.num_items
        #注意数组复制问题！
        trainUser = self.model.dataset.trainUser.copy()
        trainItem = self.model.dataset.trainItem.copy()
        keep_idx = randint_choice(len(self.model.dataset.trainUser), size=int(len(self.model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = trainUser[keep_idx]
        item_np = trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        if self.config['if_big_matrix']:
            g = self.model.dataset._split_matrix(adj_matrix)
            for fold in g:
                fold.requires_grad = False
        else:
            g = self.model.dataset._convert_sp_mat_to_sp_tensor(adj_matrix).coalesce().to(world.device)
            g.requires_grad = False
        return g

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Edge_drop_random(p_drop)
        self.augAdjMatrix2 =  self.Edge_drop_random(p_drop)



class RW_Uniform(ED_Uniform):
    def __init__(self, config, model, precal, homophily):
        super(RW_Uniform, self).__init__(config, model, precal, homophily)

    def Random_Walk(self, p_drop):
        aug_g = []
        for layer in range(self.config['num_layers']):
            aug_g.append(self.Edge_drop_random(p_drop))
        return aug_g

    # def computer(self, p_drop):
    #     aug_g = self.Random_Walk(p_drop)
    #     return self.model.view_computer(aug_g)

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Random_Walk(p_drop)
        self.augAdjMatrix2 =  self.Random_Walk(p_drop)


class SVD_Augment():
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.n_layers = config['num_layers']
        self.num_users, self.num_items = self.model.num_users, self.model.num_items

    def reconstruct_graph_computer_origin(self):
        users_emb = self.model.embedding_user.weight
        items_emb = self.model.embedding_item.weight
        embs_u = [users_emb]    
        embs_i = [items_emb]    
        for layer in range(self.n_layers):
            #propagation for user, neighbors are items
            vt_ei = self.precal.svd.svd_v_T @ embs_i[layer]
            emb_u = self.precal.svd.u_mul_s @ vt_ei
            embs_u.append(emb_u)
            #propagation for item, neighbors are users
            ut_eu = self.precal.svd.svd_u_T @ embs_u[layer]
            emb_i = self.precal.svd.v_mul_s @ ut_eu
            embs_i.append(emb_i)

        embs_u = torch.stack(embs_u, dim=1)
        light_out_user = torch.mean(embs_u, dim=1)
        embs_i = torch.stack(embs_i, dim=1)
        light_out_item = torch.mean(embs_i, dim=1)


        return light_out_user, light_out_item

    def reconstruct_graph_computer(self):
        with torch.no_grad():
            users_emb = self.model.embedding_user.weight
            items_emb = self.model.embedding_item.weight
            all_emb = torch.cat([users_emb, items_emb])
            embs = [all_emb]
            graph = self.precal.svd.u_mul_s_mul_v_T
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(graph, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
        
        return users, items




class Adaptive_Neighbor_Augment:
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.L = self.config['num_layers']
        self.epsilon = self.config['epsilon_GCLRec']
        self.w = self.config['w_GCLRec']
    
    def get_adaptive_neighbor_augment(self, embs_per_layer, batch_users, batch_pos, batch_neg, k):
        '''
        return aug_all_users, aug_all_items of selected k-th layer\n
        u'(k) = (1-𝜀)*u(k) + (𝜀(L-k)/L)*u(L) + w Σ w_uv*v(L)
        '''
        with torch.no_grad():
            pass
            #TODO
        aug_embs_k_layer = (1-self.epsilon) * embs_per_layer[k] + (self.epsilon*(self.L-k)/self.L) * embs_per_layer[self.L]
        Sigma = 0

        # low = torch.zeros_like(aug_embs_k_layer).float()
        # high = torch.ones_like(aug_embs_k_layer).float()
        # random_noise = torch.distributions.uniform.Uniform(low, high).sample()
        # noise = torch.mul(torch.sign(aug_embs_k_layer),torch.nn.functional.normalize(random_noise, dim=1))
        
        aug_embs_k_layer = aug_embs_k_layer + self.w * Sigma

        aug_user_embs_k_layer, aug_item_embs_k_layer = torch.split(aug_embs_k_layer, [self.model.num_users, self.model.num_items])
        return aug_user_embs_k_layer, aug_item_embs_k_layer
        
    def get_adaptive_neighbor_augment_batch(self, embs_per_layer, batch_users, batch_pos, batch_neg, k):
        '''
        return aug_all_users, aug_all_items of selected k-th layer\n
        u'(k) = (1-𝜀)*u(k) + (𝜀(L-k)/L)*u(L) + wΣ w_uv*v(L)
        '''

        return 
        
    def sample(self):
        '''
        sample several samples for each user or item
        '''
        return 



class Projector(torch.nn.Module):
    '''
    d --> Linear --> 2d --> BatchNorm --> ReLU --> 2d --> Linear --> d
    '''
    def __init__(self):
        super(Projector, self).__init__()

        self.input_dim = world.config['latent_dim_rec']
        self.linear1 = torch.nn.Linear(self.input_dim, 2*self.input_dim)
        self.BN = torch.nn.BatchNorm1d(2*self.input_dim)
        self.activate = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2*self.input_dim, self.input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.BN(x)
        x = self.activate(x)
        x = self.linear2(x)

        return x


# ==================== 可学习的图数据增强：学得边的权重 ====================
class Augment_Learner(torch.nn.Module):
    def __init__(self, config, Recmodel:LightGCN, precal:precalculate, homophily:Homophily, dataset:dataset):
        super(Augment_Learner, self).__init__()
        self.config = config
        self.Recmodel = Recmodel        
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.trainUser = dataset._trainUser
        self.trainItem = dataset._trainItem
        self.num_edges = len(self.trainUser)
        self.src = torch.cat([torch.tensor(self.trainUser), torch.tensor(self.trainItem)])
        self.dst = torch.cat([torch.tensor(self.trainItem), torch.tensor(self.trainUser)])
        self.edge_index = torch.tensor([list(np.append(self.trainUser, self.trainItem)), list(np.append(self.trainItem, self.trainUser))])


        self.input_dim = self.config['latent_dim_rec']
        mlp_edge_model_dim = self.config['latent_dim_rec']

        self.GNN_encoder = GNN_Encoder_edge_index(config['num_layers'], self.num_users, self.num_items)
        self.mlp_edge_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim * 2, mlp_edge_model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_edge_model_dim, 1)
        )
        # self.init_emb() TODO 
        
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if world.config['init_method'] == 'Normal':
                    torch.nn.init.normal_(m.weight.data)
                elif world.config['init_method'] == 'Xavier':
                    torch.nn.init.xavier_uniform_(m.weight.data)
                else:
                    torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self):
        ''''
        返回增强后的边权重
        '''           

        users_emb, items_emb = self.GNN_encoder.forward(self.Recmodel.pyg_data())
        nodes_emb = torch.cat([users_emb, items_emb])

        emb_src = nodes_emb[self.src]
        emb_dst = nodes_emb[self.dst]
        #只改变原有边的权重（邻接矩阵中的1变成0~1的实数）
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)
        # edge_logits1, edge_logits2 = torch.split(edge_logits, [self.num_edges, self.num_edges])
        # edge_logits = (edge_logits1 + edge_logits2) * 0.5

        with torch.no_grad():
            users_emb0 = self.Recmodel.embedding_user.weight
            items_emb0 = self.Recmodel.embedding_item.weight
            x = torch.cat([users_emb0, items_emb0])
            # graph = self.Recmodel.Graph

        data_aug = torch_geometric.data.Data(x=x, edge_index=self.edge_index.contiguous(), edge_attr=edge_logits)#TODO detach

        #TODO 将edge_index格式的数据再构建为邻接矩阵并归一化
        # aug_adj_mat = self.get_graph(edge_logits.cpu().detach().squeeze(), self.trainUser, self.trainItem, self.num_users, self.num_items)

        return data_aug.detach()
    
    # def get_graph(self, edge_logits, trainUser, trainItem, n_user, m_item):
    #     UserItemNet = csr_matrix((edge_logits, (trainUser, trainItem)), shape=(n_user, m_item))

    #     adj_mat = sp.dok_matrix((n_user + m_item, n_user + m_item), dtype=np.float32)
    #     adj_mat = adj_mat.tolil()
    #     R = UserItemNet.tolil()
    #     #此处会显存爆炸
    #     adj_mat[:n_user, n_user:] = R
    #     adj_mat[n_user:, :n_user] = R.T
    #     adj_mat = adj_mat.todok()
    #     # adj_mat = adj_mat + sp.eye(adj_mat.shape[0]) TODO 无自连接
        
    #     rowsum = np.array(adj_mat.sum(axis=1))
    #     d_inv = np.power(rowsum, -0.5).flatten()
    #     d_inv[np.isinf(d_inv)] = 0.
    #     d_mat = sp.diags(d_inv)#对角阵
    #     #TODO 不再归一化？
    #     norm_adj = adj_mat
    #     # norm_adj = d_mat.dot(adj_mat)
    #     # norm_adj = norm_adj.dot(d_mat)
    #     norm_adj = norm_adj.tocsr()

    #     Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
    #     Graph = Graph.coalesce().to(world.device)

    #     return Graph
    
    # def _convert_sp_mat_to_sp_tensor(self, X):
    #     coo = X.tocoo().astype(np.float32)
    #     row = torch.Tensor(coo.row).long()
    #     col = torch.Tensor(coo.col).long()
    #     index = torch.stack([row, col])
    #     data = torch.FloatTensor(coo.data)
    #     return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
