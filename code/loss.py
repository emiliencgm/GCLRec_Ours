"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
from precalcul import precalculate
import torch
import torch.nn.functional as F
import world
from augment import Homophily
import numpy as np
import math
#=============================================================BPR loss============================================================#
class BPR_loss():
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precalculate = precalculate

    def bpr_loss(self, batch_user, batch_pos, batch_neg):
        loss, reg_loss = self.model.bpr_loss(batch_user, batch_pos, batch_neg)
        reg_loss = reg_loss * self.config['weight_decay']
        loss = loss + reg_loss
        loss = loss/self.config['batch_size']
        return loss

#=============================================================BPR + CL loss============================================================#
class BPR_Contrast_loss(BPR_loss):
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        super(BPR_Contrast_loss, self).__init__(config, model, precalculate, homophily)
        self.tau = config['temp_tau']

    def bpr_contrast_loss(self, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_user, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2):

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))#TODO softplus
        bprloss = (loss + reg_loss * self.config['weight_decay'])/self.config['batch_size']
        
        contrastloss = self.info_nce_loss_overall(aug_users1[batch_user], aug_users2[batch_user], aug_users2) \
                        + self.info_nce_loss_overall(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2)
        return bprloss + self.config['lambda1']*contrastloss

    def info_nce_loss_overall(self, z1, z2, z_all):
        '''
        z1--z2: pos,  z_all: neg\n
        return: InfoNCEloss
        '''
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        all_sim = f(self.sim(z1, z_all))
        positive_pairs = (between_sim)
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))#TODO softplus
        #print('positive_pairs / negative_pairs',max(positive_pairs / negative_pairs))
        loss = loss/self.config['batch_size']
        return loss


    def sim(self, z1: torch.Tensor, z2: torch.Tensor, mode='inner_product'):#TODO
        '''
        计算一个z1和一个z2两个向量的相似度/或者一个z1和多个z2的各自相似度。
        即两个输入的向量数（行数）可能不同。
        '''
        if mode == 'inner_product':
            if z1.size()[0] == z2.size()[0]:
                #return F.cosine_similarity(z1,z2)
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                return torch.sum(torch.mul(z1,z2) ,dim=1)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())
        elif mode == 'cos':
            if z1.size()[0] == z2.size()[0]:
                return F.cosine_similarity(z1,z2)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())
#=============================================================Softmax loss============================================================#
class Softmax_loss():
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precalculate = precalculate
        self.tau = config['temp_tau']

    def softmax_loss(self, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_user, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2):

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        # pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        # neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), torch.transpose(neg_emb, 0, 1)).squeeze(dim=1)
        # ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim=1)

        # numerator = torch.exp(pos_ratings / self.tau)
        # denominator = torch.sum(torch.exp(ratings / self.tau), dim=1)

        softmax_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.config['batch_size']
        reg_loss = self.config['weight_decay'] * regularizer

        return softmax_loss + reg_loss

#=============================================================BC loss============================================================#       
class BC_loss():
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precalculate = precalculate
        self.tau1 = config['temp_tau']
        self.tau2 = config['temp_tau_pop']
        self.decay = self.config['weight_decay']
        self.batch_size = self.config['batch_size']
        self.alpha = self.config['alpha']#BC loss下的alpha与Adaptive下的alpha不同，是用于对pop_loss和主bc_loss加权用的。

    def bc_loss(self, batch_target, batch_pos, batch_negs, mode):
        #只使用正样本（一个user的正样本往往是其他user的负样本），无需负采样
        users_pop = torch.tensor(self.precalculate.popularity.user_pop_degree_label).to(world.device)[batch_target]
        pos_items_pop = torch.tensor(self.precalculate.popularity.item_pop_degree_label).to(world.device)[batch_pos]
        bc_loss, pop_loss, reg_pop_emb_loss, reg_pop_loss, reg_emb_loss = self.calculate_loss(batch_target, batch_pos, users_pop, pos_items_pop)
        if mode == 'only_bc':
            loss = bc_loss + reg_emb_loss
        elif mode == 'pop_bc':
            loss = bc_loss + pop_loss + reg_pop_emb_loss
        elif mode =='only_pop':
            loss = pop_loss + reg_pop_loss
        return loss
    
    #From BC loss
    def calculate_loss(self, users, pos_items, users_pop, pos_items_pop):

        # popularity branch
        users_pop_emb = self.model.embed_user_pop(users_pop)
        pos_pop_emb = self.model.embed_item_pop(pos_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)

        ratings = torch.matmul(users_pop_emb, torch.transpose(pos_pop_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.alpha * torch.mean(torch.negative(torch.log(numerator/denominator)))
        #loss2 = self.alpha * torch.sum(torch.negative(torch.log(numerator/denominator)))

        # main bc branch
        all_users, all_items = self.model.computer()

        userEmb0 = self.model.embedding_user(users)
        posEmb0 = self.model.embedding_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+\
                                (1-torch.sigmoid(pos_ratings_margin)))
        #ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+\
        #                        (torch.zeros_like(pos_ratings_margin)))
        
        numerator = torch.exp(ratings_diag / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        loss1 = (1-self.alpha) * torch.mean(torch.negative(torch.log(numerator/denominator)))
        #loss1 = (1-self.alpha) * torch.sum(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size
        #regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        #regularizer1 = regularizer1

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        #regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_emb) ** 2
        #regularizer2  = regularizer2
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm


#=============================================================Myself Adaptive loss============================================================#
class MLP(torch.nn.Module):
    def __init__(self, in_dim):
        super(MLP, self).__init__()

        #1-->2-->2-->1
        self.linear1=torch.nn.Linear(in_dim,3*in_dim)
        self.activation1=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(3*in_dim, in_dim)
        self.activation2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(in_dim,1)
        #TODO from CV projector:
        self.linear = torch.nn.Linear(in_dim,4*in_dim)
        self.BatchNorm = torch.nn.BatchNorm1d(4*in_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = torch.nn.ReLU()
        self.linear_out = torch.nn.Linear(4*in_dim, 1)

    def forward(self, x):
        #TODO Architecture suboptimal
        # x = self.linear1(x)
        # x = self.activation1(x)
        # x = self.linear2(x)
        # x = self.activation2(x)
        # x = self.linear3(x)
        x = self.linear(x)
        # x = self.BatchNorm(x)
        x = self.activation(x)
        x = self.linear_out(x)
        x = torch.sigmoid(x)
        return x



class Adaptive_softmax_loss(torch.nn.Module):
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        super(Adaptive_softmax_loss, self).__init__()

        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.tau = config['temp_tau']
        self.alpha = config['alpha']
        self.f = lambda x: torch.exp(x / self.tau)
        self.MLP_model = MLP(5+2*0).to(world.device)

    def adaptive_softmax_loss(self, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_user, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2):
        
        reg = (0.5 * torch.norm(userEmb0) ** 2 + len(batch_pos) * 0.5 * torch.norm(posEmb0) ** 2)/len(batch_pos)
        loss1 = self.calculate_loss(users_emb, pos_emb, neg_emb, batch_user, batch_pos, self.config['adaptive_method'], self.config['centroid_mode'])
        #loss1 = self.calculate_loss(users_emb, pos_emb, neg_emb, None, None, None, None)
        if not (aug_users1 is None):
            loss2 = self.calculate_loss(aug_users1[batch_user], aug_users2[batch_user], aug_users2, None, None, None, None)
            loss3 = self.calculate_loss(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2, None, None, None, None)
            loss = self.config['weight_decay']*reg + loss1 + self.config['lambda1']*(loss2 + loss3)
        else:
            loss = self.config['weight_decay']*reg + loss1
        # loss_homophily = self.calculate_loss_homophily(users_emb, pos_emb)
        # loss = loss + loss_homophily
        
        return loss


    def calculate_loss(self, batch_target_emb, batch_pos_emb, batch_negs_emb, batch_target, batch_pos, method, mode):
        '''
        input : embeddings, not index.
        '''

        users_emb = batch_target_emb
        pos_emb = batch_pos_emb

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        # ratings_margin = self.get_coef_adaptive_all(batch_target_emb.detach(), batch_pos_emb.detach())
        # ratings = torch.cos(torch.arccos(torch.clamp(ratings,-1+1e-7,1-1e-7))+ratings_margin)
        ratings_diag = torch.diag(ratings)
        if not (method is None):
            #Adaptive coef between User and Item
            pos_ratings_margin = self.get_coef_adaptive(batch_target, batch_pos, method=method, mode=mode)
            theta = torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))
            M = torch.arccos(torch.clamp(pos_ratings_margin,-1+1e-7,1-1e-7))
            # M = torch.clamp(M, torch.zeros_like(M), math.pi-theta)
            ratings_diag = torch.cos(theta + M)
            # ratings_diag = ratings_diag * pos_ratings_margin
            #reliable / important ==> big margin ==> small theta ==> big simi between u,i 
        else:
            # pos_ratings_margin = self.get_coef_adaptive_aug(batch_target_emb, batch_pos_emb)
            # ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+torch.arccos(torch.clamp(pos_ratings_margin,-1+1e-7,1-1e-7)))
            pass
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        #loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        loss = torch.mean(torch.negative((2*self.alpha * torch.log(numerator) -  2*(1-self.alpha) * torch.log(denominator))))
        #TODO trying torch.nn.functional.softplus 
        
        return loss

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def calculate_loss_homophily(self, embs1, embs2):
        batch_prob1, batch_prob2 = self.homophily.get_homophily_batch_any(embs1, embs2)
        loss_homophily = 0.5*(torch.nn.functional.kl_div(batch_prob1.log(), batch_prob2, reduction='batchmean') + torch.nn.functional.kl_div(batch_prob2.log(), batch_prob1, reduction='batchmean'))
        return loss_homophily


    def get_popdegree(self, batch_user, batch_pos_item):
        with torch.no_grad():
            pop_user = torch.tensor(self.precal.popularity.user_pop_degree_label).to(world.device)[batch_user]
            pop_item = torch.tensor(self.precal.popularity.item_pop_degree_label).to(world.device)[batch_pos_item]
        return pop_user, pop_item

    def get_homophily(self, batch_user, batch_pos_item):
        with torch.no_grad():
            batch_user_prob, batch_item_prob = self.homophily.get_homophily_batch(batch_user, batch_pos_item)
            batch_weight = torch.sum(torch.mul(batch_user_prob, batch_item_prob) ,dim=1)
        return batch_weight
    
    def get_centroid(self, batch_user, batch_pos_item, centroid='eigenvector', aggr='mean', mode='GCA'):
        with torch.no_grad():
            batch_weight = self.precal.centroid.cal_centroid_weights_batch(batch_user, batch_pos_item, centroid=centroid, aggr=aggr, mode=mode)
        return batch_weight
    
    def get_commonNeighbor(self, batch_user, batch_pos_item):
        with torch.no_grad():
            n_users = self.model.num_users
            csr_matrix_CN_simi = self.precal.common_neighbor.CN_simi_mat_sp
            batch_user, batch_pos_item = np.array(batch_user.cpu()), np.array(batch_pos_item.cpu())
            batch_weight1 = csr_matrix_CN_simi[batch_user, batch_pos_item+n_users]
            batch_weight2 = csr_matrix_CN_simi[batch_pos_item+n_users, batch_user]
            batch_weight1 = torch.tensor(np.array(batch_weight1).reshape((-1,))).to(world.device)
            batch_weight2 = torch.tensor(np.array(batch_weight2).reshape((-1,))).to(world.device)
        return batch_weight1, batch_weight2

    def get_embs(self, batch_user, batch_pos_item):
        with torch.no_grad():
            batch_weight_user, batch_weight_item = self.model.embedding_user(batch_user), self.model.embedding_item(batch_pos_item)
        return batch_weight_user, batch_weight_item

    def get_mlp_input(self, features):
        '''
        features = [tensor, tensor, ...]
        '''
        U = features[0].unsqueeze(0)
        for i in range(1,len(features)):
            U = torch.cat((U, features[i].unsqueeze(0)), dim=0)
        return U.T


    def get_coef_adaptive(self, batch_user, batch_pos_item, method='mlp', mode='eigenvector'):
        '''
        input: index batch_user & batch_pos_item\n
        return tensor([adaptive coefficient of u_n-i_n])\n
        the bigger, the more reliable, the more important
        '''
        if method == 'homophily':
            batch_weight = self.get_homophily(batch_user, batch_pos_item)
            batch_weight = 1. * batch_weight 

        elif method == 'centroid':
            batch_weight = self.get_centroid(batch_user, batch_pos_item, centroid=mode, aggr='mean', mode='GCA')
            batch_weight = 1. * batch_weight

        elif method == 'commonNeighbor':
            batch_weight1, batch_weight2 = self.get_commonNeighbor(batch_user, batch_pos_item)
            batch_weight = (batch_weight1 + batch_weight2)*0.5
            batch_weight = 1. * batch_weight

        elif method == 'mlp':
            batch_weight_emb_user, batch_weight_emb_item = self.get_embs(batch_user, batch_pos_item)
            batch_weight_pop_user, batch_weight_pop_item = self.get_popdegree(batch_user, batch_pos_item)
            batch_weight_pop_user, batch_weight_pop_item = torch.log(batch_weight_pop_user), torch.log(batch_weight_pop_item)#TODO problem of grandeur
            #batch_weight_homophily = self.get_homophily(batch_user, batch_pos_item)
            batch_weight_centroid = self.get_centroid(batch_user, batch_pos_item, centroid=mode, aggr='mean', mode='GCA')
            batch_weight_commonNeighbor1, batch_weight_commonNeighbor2 = self.get_commonNeighbor(batch_user, batch_pos_item)
            features = [batch_weight_pop_user, batch_weight_pop_item, batch_weight_centroid, batch_weight_commonNeighbor1, batch_weight_commonNeighbor2]
            # features = []
            # with torch.no_grad():
            #     for i in range(self.config['latent_dim_rec']):
            #         features.append(batch_weight_emb_user[:,i])
            #     for i in range(self.config['latent_dim_rec']):
            #         features.append(batch_weight_emb_item[:,i])
            
            batch_weight = self.get_mlp_input(features)
            batch_weight = self.MLP_model(batch_weight)

        else:
            batch_weight = None
            raise TypeError('adaptive method not implemented')

        # batch_weight = torch.sigmoid(batch_weight)#TODO
        
        return batch_weight


    def get_coef_adaptive_aug(self, aug1, aug2):
        return 0

    def get_coef_adaptive_all(self, embs1, embs2):
        '''
        backpropagation only update mlp, not embs, so embs are detached.
        '''
        #TODO
        num_row = embs1.shape[0]
        num_col = embs2.shape[0]
        batch_weight = torch.zeros((num_row, num_col))
        features = torch.zeros(1,128).to(world.device)
        for row in range(num_row):
            for col in range(row, num_col):
                features = torch.cat( (features, torch.cat((embs1[row].unsqueeze(0), embs2[col].unsqueeze(0)), dim=1)), dim=0 )
        mlp_output = self.MLP_model(features)
        i = 1
        for row in range(num_row):
            for col in range(row, num_col):
                batch_weight[row,col] = mlp_output[i]
                batch_weight[col,row] = mlp_output[i]
                i = i + 1

        # for row in range(num_row):
        #     for col in range(row, num_col):
        #         mlp_input = torch.cat((embs1[row], embs2[col]))
        #         output = self.MLP_model(mlp_input)
        #         batch_weight[row,col] = output
        #         batch_weight[col,row] = output

        return batch_weight
#=============================================================Debiased Contrastive loss============================================================#
class Debiased_Contrastive_loss():
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.tau = config['temp_tau']
        self.alpha = config['alpha']
        self.f = lambda x: torch.exp(x / self.tau)

    def debiased_contrastive_loss(self, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_user, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2):
        '''
        只计算u-i, 不使用数据增强后的u-u'
        '''
        reg = (0.5 * torch.norm(userEmb0) ** 2 + len(batch_pos) * 0.5 * torch.norm(posEmb0) ** 2)/len(batch_pos)
        loss_ui = self.calculate_loss(users_emb, pos_emb)
        loss = self.config['weight_decay']*reg + loss_ui
        
        return loss


    def calculate_loss(self, batch_target_emb, batch_pos_emb):
        '''
        input : embeddings, not index.
        '''

        users_emb = batch_target_emb
        pos_emb = batch_pos_emb

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)        
        
        #TODO 在一次计算中：pos-pairs:{u-i}, neg-pairs:{u-u', u-i'}
        batch_size = world.config['batch_size']
        tau_plus = world.config['tau_plus']
        temperature = world.config['temp_tau']
        N = batch_size * 2 - 2
        pos = torch.exp(torch.sum(users_emb * pos_emb, dim=-1) / temperature)
        # #----------adding--BC----------
        # ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        # pos_adaptive = torch.diag(ratings).squeeze()
        # pos_margin = self.get_coef_adaptive(batch_target, batch_pos, method=method, mode=mode).squeeze()
        # theta = torch.arccos(torch.clamp(pos_adaptive,-1+1e-7,1-1e-7))
        # M = torch.arccos(torch.clamp(pos_margin,-1+1e-7,1-1e-7))
        # # M = torch.clamp(M, torch.zeros_like(M), math.pi-theta)
        # pos_adaptive = torch.cos(theta + M)
        # pos_adaptive = torch.exp(pos_adaptive / temperature)
        # pos_adaptive = torch.cat([pos_adaptive, pos_adaptive], dim=0)
        # #----------adding--BC----------
        pos = torch.cat([pos, pos], dim=0)
        out = torch.cat([users_emb, pos_emb], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        loss = (- torch.log(1.0*pos / (pos + Ng) )).mean()
        
        return loss

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

#=============================================================Causal Popularity Bias BPR loss============================================================#
class Causal_popularity_BPR_loss():
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precalculate = precalculate
        self.gamma = config['pop_gamma']

    def causal_popularity_bpr_loss(self, batch_user, batch_pos, batch_neg):
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, _) = self.model.getEmbedding(batch_user.long(), batch_pos.long(), batch_neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))
        reg_loss = reg_loss * self.config['weight_decay']

        pos_items_pop = torch.tensor(self.precalculate.popularity.item_pop_degree_label)[batch_pos].to(world.device)
        neg_items_pop = torch.tensor(self.precalculate.popularity.item_pop_degree_label)[batch_neg].to(world.device)
        # pos_pop_emb = self.model.embed_item_pop(pos_items_pop)
        # neg_pop_emb = self.model.embed_item_pop(neg_items_pop)
        norm_pos_items_pop = pos_items_pop / self.precalculate.popularity.item_pop_sum
        norm_neg_items_pop = neg_items_pop / self.precalculate.popularity.item_pop_sum
        norm_pos_items_pop = norm_pos_items_pop ** self.gamma
        norm_neg_items_pop = norm_neg_items_pop ** self.gamma

        elu = torch.nn.ELU()

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        pos_scores = elu(pos_scores) + 1.
        pos_scores = torch.mul(pos_scores, norm_pos_items_pop)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        neg_scores = elu(neg_scores) + 1.
        neg_scores = torch.mul(neg_scores, norm_neg_items_pop)

        logSigmoid = torch.nn.LogSigmoid()
        
        loss = torch.sum(-logSigmoid((pos_scores - neg_scores)))

        loss = (loss + reg_loss)/self.config['batch_size']

        return loss

