# model_light_gcrec.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility.losses
import utility.tools
from dgl.nn.pytorch import GraphConv
from .ViewLearner import ViewLearner

class LightEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LightEncoder, self).__init__()
        self.gcn = GraphConv(in_dim, out_dim, norm='both', allow_zero_in_degree=True)
        self.res = nn.Linear(in_dim, out_dim)

    def forward(self, g, x):
        h = self.gcn(g, x)
        return h + self.res(x)


class GCRec(nn.Module):
    def __init__(self, config, dataset, user_g, item_g, device):
        super(GCRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(config.reg_lambda)
        self.ssl_lambda = float(config.ssl_lambda)
        self.intra_lambda = float(config.intra_lambda)
        self.temperature = float(config.temperature)

        self.view_learner = ViewLearner(input_dim=config.dim, output_dim=config.dim)
        self.user_embedding = nn.Embedding(dataset.num_users, config.dim)
        self.item_embedding = nn.Embedding(dataset.num_items, config.dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.Graph = utility.tools.convert_sp_mat_to_sp_tensor(
            dataset.sparse_adjacency_matrix()
        ).coalesce().to(device)

        self.activation = nn.Sigmoid()
        self.uu_graph = user_g
        self.ii_graph = item_g

        self.user_encoder = nn.ModuleList([LightEncoder(config.dim, config.dim) for _ in user_g])
        self.item_encoder = nn.ModuleList([LightEncoder(config.dim, config.dim) for _ in item_g])

    def aggregate(self):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings = []
        for _ in range(int(self.config.GCN_layer)):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embeddings.append(all_emb)
        final = torch.stack(embeddings, dim=1).mean(1)
        return torch.split(final, [self.dataset.num_users, self.dataset.num_items])

    def forward(self, user, positive, negative, epoch=None):
        user_emb, item_emb = self.aggregate()

        hete_user_embs = [enc(self.uu_graph[i], self.user_embedding.weight) for i, enc in enumerate(self.user_encoder)]
        hete_item_embs = [enc(self.ii_graph[i], self.item_embedding.weight) for i, enc in enumerate(self.item_encoder)]

        user_view = torch.mean(torch.stack(hete_user_embs), dim=0)
        item_view = torch.mean(torch.stack(hete_item_embs), dim=0)

        user_intra = utility.losses.get_InfoNCE_loss(hete_user_embs[0][user], hete_user_embs[1][user], self.temperature)
        item_intra = utility.losses.get_InfoNCE_loss(hete_item_embs[0][positive], hete_item_embs[1][positive], self.temperature)
        intra_loss = self.intra_lambda * (user_intra + item_intra)

        user_ssl = utility.losses.get_InfoNCE_loss(user_emb[user], user_view[user], self.temperature)
        item_ssl = utility.losses.get_InfoNCE_loss(item_emb[positive], item_view[positive], self.temperature)
        ssl_loss = self.ssl_lambda * (user_ssl + item_ssl)

        pos = item_emb[positive]
        neg = item_emb[negative]
        user_e = user_emb[user]
        bpr_loss = utility.losses.get_bpr_loss(user_e, pos, neg)
        reg_loss = self.reg_lambda * utility.losses.get_reg_loss(
            self.user_embedding(user), self.item_embedding(positive), self.item_embedding(negative)
        )

        return [bpr_loss, reg_loss, ssl_loss, intra_loss]

    def get_rating_for_test(self, user):
        u, i = self.aggregate()
        return self.activation(torch.matmul(u[user], i.t()))

    def get_embedding(self):
        return self.aggregate()
