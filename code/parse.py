"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go SGL")
    #WandB
    parser.add_argument('--project', type=str, default='project', help="wandb project")
    parser.add_argument('--name', type=str, default='name', help="wandb name")   
    parser.add_argument('--notes', type=str, default='-', help="wandb notes")   
    parser.add_argument('--tag', nargs='+', help='wandb tags')
    parser.add_argument('--group', type=str, default='-', help="wandb group") 
    parser.add_argument('--job_type', type=str, default='-', help="wandb job_type") 
    #Hyperparameters===========================================================================================================================================
    parser.add_argument('--temp_tau', type=float, default=0.2, help="tau in InfoNCEloss")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha for balancing loss terms OR weighting pop_loss & bc_loss in BC loss")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay == lambda2")
    parser.add_argument('--lambda1', type=float, default=0.1, help="lambda1 == coef of Contrstloss")
    parser.add_argument('--n_cluster', type=int, default=10, help="Num of clutsers while using K-Means for homophily similarity calculation")
    parser.add_argument('--sigma_gausse', type=float, default=1., help="Sigma in the Gaussian mixture distribution")
    parser.add_argument('--epsilon_GCLRec', type=float, default=0.1, help="epsilon for controling augmentation in GCLRec")
    parser.add_argument('--w_GCLRec', type=float, default=0.1, help="w for controling augmentation in GCLRec")
    parser.add_argument('--k_aug', type=int, default=0, help="use k-th layer for augmentation in GCLRec, k in [0,1,2,... L]")
    #===========================================================================================================================================

    parser.add_argument('--early_stop_steps', type=int, default=30, help="early stop steps")
    parser.add_argument('--latent_dim_rec', type=int, default=64, help="latent dim for rec")
    parser.add_argument('--num_layers', type=int, default=3, help="num layers of LightGCN")
    parser.add_argument('--if_pretrain', type=int, default=0, help="whether use pretrained Embedding")   
    parser.add_argument('--if_load_embedding', type=int, default=0, help="whether load trained embedding")
    parser.add_argument('--if_tensorboard', type=int, default=1, help="whether use tensorboardX")
    parser.add_argument('--epochs', type=int, default=1000, help="training epochs")
    parser.add_argument('--if_multicore', type=int, default=1, help="whether use multicores in Test")
    parser.add_argument('--batch_size', type=int, default=2048, help="batch size in BPR_Contrast_Train")    
    parser.add_argument('--topks', nargs='?', default='[20]', help="topks [@20] for test")
    parser.add_argument('--test_u_batch_size', type=int, default=2048, help="users batch size for test")
    parser.add_argument('--pop_group', type=int, default=10, help="Num of groups of Popularity")
    parser.add_argument('--if_big_matrix', type=int, default=0, help="whether the adj matrix is big, and then use matrix n_fold split")
    parser.add_argument('--n_fold', type=int, default=2, help="split the matrix to n_fold")
    parser.add_argument('--cuda', type=str, default='0', help="cuda id")
    parser.add_argument('--visual_epoch', type=int, default=1, help="visualize every tsne_epoch")
    parser.add_argument('--if_double_label', type=int, default=1, help="whether use item categories label along with popularity group")
    parser.add_argument('--if_tsne', type=int, default=1, help="whether use t-SNE")
    parser.add_argument('--tsne_group', nargs='?', default='[0, 9]', help="groups [0, 9] for t-SNE")    
    parser.add_argument('--tsne_points', type=int, default=2000, help="Num of points of users/items in t-SNE")
    parser.add_argument('--if_visual', type=int, default=0, help="whether use visualization, i.e. t_sne, double_label")    

    #Architecture===========================================================================================================================================
    parser.add_argument('--encoder', type=str, default='LightGCN', help="Now available:\n\
                                                                    ###LightGCN\n\
                                                                    ###LightGCN_PyG: PyG implementation (SimpleConv) of LightGCN")
    parser.add_argument('--dataset', type=str, default='yelp2018', help="dataset:[yelp2018,  gawalla, ifashion, amazon-book,  MIND]") 
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--loss', type=str, default='Adaptive', help="loss function: BPR, BPR_Contrast, BC, Adaptive, Causal_pop, DCL")
    parser.add_argument('--augment', type=str, default='No', help="Augmentation: No, ED, RW, SVD, Adaptive, Learner")    
    parser.add_argument('--centroid_mode', type=str, default='eigenvector', help="Centroid mode: degree, pagerank, eigenvector")
    parser.add_argument('--commonNeighbor_mode', type=str, default='SC', help="Common Neighbor mode: JS, SC, CN, LHN")
    parser.add_argument('--adaptive_method', type=str, default='None', help="Adaptive coef method: centroid, commonNeighbor, homophily, mlp")
    parser.add_argument('--init_method', type=str, default='Normal', help="UI embeddings init method: Xavier or Normal")
    parser.add_argument('--perplexity', type=int, default=50, help="perplexity for T-SNE")
    parser.add_argument('--if_projector', type=int, default=0, help="whether use Projector(a 2-Layer MLP) for augmented-view embedding")
    parser.add_argument('--comment', type=str, default='_', help="comment for the experiment")
    parser.add_argument('--if_valid', type=int, default=0, help="whether use validtion set")
    #===========================================================================================================================================

    return parser.parse_args()