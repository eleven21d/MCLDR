# model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility.losses as losses
from model.denoise_encoder import DenoiseEncoder

class MCLKR(nn.Module):
    """
    Use with the light GCRec：
    - forward(): First call base_model.forward() to obtain the original loss_list,
    then calculate the cross-view InfoNCE between the denoised view and the meta-path view,
    and optionally recompute BPR with fused embeddings.
    - Evaluation: get_rating_for_test()/get_embedding() returns the fused embedding results.

    """

    def __init__(self, base_model,
                 denoise_edge_path,
                 denoise_embed_dim=None,
                 denoise_num_layers=2,
                 denoise_lambda: float = 0.2,
                 denoise_temperature: float = None,
                 rel_weight=None,
                 device=None,
                 fusion_alpha: float = 0.1,     # fusion α
                 replace_bpr: bool = True      # fusion bpr(optional)
                 ):
        super().__init__()
        self.base_model = base_model
        self.device = device if device is not None else getattr(base_model, "device", torch.device("cpu"))

        if denoise_embed_dim is None:
            denoise_embed_dim = int(getattr(base_model.config, "dim", 64))

        self.num_users = int(base_model.dataset.num_users)
        self.num_items = int(base_model.dataset.num_items)

        self.denoise_encoder = DenoiseEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            embed_dim=denoise_embed_dim,
            edge_path=denoise_edge_path,
            rel_weight=rel_weight,
            num_layers=denoise_num_layers,
            device=self.device
        ).to(self.device)

        self.tau = float(denoise_temperature if denoise_temperature is not None
                         else getattr(base_model, "temperature", 0.2))
        self.denoise_lambda = float(denoise_lambda)

        self.fusion_alpha = float(fusion_alpha)
        self.replace_bpr = bool(replace_bpr)

        try:
            self.base_model.to(self.device)
        except Exception:
            pass

    def _cross_view_infonce(self, z_meta_all, z_denoise_all, indices):
        a = z_meta_all[indices.long()]
        b = z_denoise_all[indices.long()]
        try:
            return losses.get_InfoNCE_loss(a, b, self.tau)
        except Exception:
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
            logits = (a @ b.t()) / self.tau
            pos = torch.diag(logits)
            return -(pos - torch.logsumexp(logits, dim=1)).mean()

    def _fuse_all_embeds(self, users_meta, items_meta, users_dn, items_dn):
        alpha = self.fusion_alpha
        users_fused = (users_meta + alpha * users_dn) / (1.0 + alpha)
        items_fused = (items_meta + alpha * items_dn) / (1.0 + alpha)
        return users_fused, items_fused

    def forward(self, user, positive, negative, epoch=None):
        # 1) origin loss(reg+intral+ssl)
        self.base_model.train() if self.training else self.base_model.eval()
        loss_list = self.base_model.forward(user, positive, negative, epoch)

        # 2) base embedding
        users_meta_all, items_meta_all = self.base_model.aggregate()  # (U,D), (I,D)
        all_meta = torch.cat([users_meta_all, items_meta_all], dim=0)  # (U+I,D)

        # denoise embedding
        dn_users_all, dn_items_all = self.denoise_encoder()           # (U,D), (I,D)
        all_dn = torch.cat([dn_users_all, dn_items_all], dim=0)

        # L_denoise
        user_cl = self._cross_view_infonce(all_meta, all_dn, user)
        item_cl = self._cross_view_infonce(
            all_meta, all_dn, positive + self.num_users
        )
        denoise_cl = self.denoise_lambda * (user_cl + item_cl)
        loss_list.append(denoise_cl)

        # fusion BPR(optional)
        if self.replace_bpr:
            users_fused, items_fused = self._fuse_all_embeds(users_meta_all, items_meta_all,
                                                             dn_users_all, dn_items_all)
            u = users_fused[user.long()]
            pos = items_fused[positive.long()]
            neg = items_fused[negative.long()]
            bpr_fused = losses.get_bpr_loss(u, pos, neg)
            loss_list[0] = bpr_fused  # fusion BPR

        return loss_list

    def get_rating_for_test(self, user):
        users_meta_all, items_meta_all = self.base_model.aggregate()
        dn_users_all, dn_items_all = self.denoise_encoder()
        users_fused, items_fused = self._fuse_all_embeds(users_meta_all, items_meta_all,
                                                         dn_users_all, dn_items_all)
        rating = torch.matmul(users_fused[user.long()], items_fused.t())
        return torch.sigmoid(rating)

    def get_embedding(self):
        users_meta_all, items_meta_all = self.base_model.aggregate()
        dn_users_all, dn_items_all = self.denoise_encoder()
        users_fused, items_fused = self._fuse_all_embeds(users_meta_all, items_meta_all,
                                                         dn_users_all, dn_items_all)
        return users_fused, items_fused
