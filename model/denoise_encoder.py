# model/denoise_encoder.py
import torch
import torch.nn as nn
from collections import defaultdict
from torch_sparse import SparseTensor

class DenoiseEncoder(nn.Module):
    def __init__(self, num_users=None, num_items=None, embed_dim=64, edge_path=None, rel_weight=None, num_layers=2, device=None):
        super().__init__()
        self.edge_path = edge_path
        self.device = device or torch.device("cpu")
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.rel_weight = rel_weight or {}

        # If num_users/num_items provided, compute total nodes, else infer later
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = (num_users + num_items) if (num_users is not None and num_items is not None) else None

        # We'll initialize embeddings lazily after we know total_nodes
        self.embedding = None

        # Preload edges
        self.edges_by_rel, self.max_id = self._load_edges()

        if self.total_nodes is None:
            self.total_nodes = self.max_id + 1

        # initialize embedding now
        self.embedding = nn.Embedding(self.total_nodes, self.embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def _load_edges(self):
        edges_by_rel = defaultdict(list)
        max_id = -1
        with open(self.edge_path, 'r') as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) < 2:
                    continue
                if len(parts) == 2:
                    s,d = int(parts[0]), int(parts[1])
                    rel = ''
                else:
                    s,d,rel = int(parts[0]), int(parts[1]), parts[2]
                edges_by_rel[rel].append((s,d))
                if s > max_id: max_id = s
                if d > max_id: max_id = d
        return edges_by_rel, max_id

    def forward(self):
        x = self.embedding.weight.to(self.device)
        all_embs = [x]

        # For each layer, aggregate by summing relation-wise message passing
        for _ in range(self.num_layers):
            agg = torch.zeros_like(x).to(self.device)
            for rel, edges in self.edges_by_rel.items():
                if len(edges) == 0:
                    continue
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)  # shape [2, E]
                # make sure indices < total_nodes
                if int(edge_index.max()) >= self.total_nodes:
                    raise RuntimeError(f"Edge id {int(edge_index.max())} >= total_nodes {self.total_nodes}. Check offsets.")
                # SparseTensor expects [row, col] with shape (N,N)
                sp = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(self.total_nodes, self.total_nodes))
                # message = sp @ x  (torch_sparse SparseTensor matmul)
                # convert to torch dense result using sp.matmul
                msg = sp.matmul(x)   # returns (N, D)
                w = self.rel_weight.get(rel, 1.0)
                agg = agg + w * msg
            # normalize by number of relations to keep scale stable
            if len(self.edges_by_rel) > 0:
                agg = agg / max(1, len(self.edges_by_rel))
            x = agg
            all_embs.append(x)

        z = torch.stack(all_embs, dim=1).mean(1)
        users_emb = z[:self.num_users] if self.num_users is not None else z
        items_emb = z[self.num_users:self.num_users + self.num_items] if (self.num_users is not None and self.num_items is not None) else z
        return users_emb, items_emb
