import torch
import torch.nn as nn
from .towers import UserTower, ItemTower

class TwoTowerModel(nn.Module):
    """
    Core Retrieval Architecture: Two-Tower (Dual Encoder).
    
    Concept:
    - Biểu diễn User và Item trong cùng một không gian vector (Latent Space).
    - Độ tương đồng (Similarity) được tính bằng Dot Product (hoặc Cosine Similarity).
    
    Key Features:
    - Weight Sharing: Chia sẻ Embedding Layer của Item ID giữa User Tower (History) và Item Tower (Target) để tiết kiệm VRAM (~6GB).
    - Learnable Temperature: Tự động điều chỉnh độ sắc nhọn của phân phối xác suất trong Contrastive Loss.
    """
    def __init__(
        self,
        user_tower_config: dict,
        item_tower_config: dict,
        temperature: float = 0.07,
        use_learnable_temp: bool = True,
    ):
        super().__init__()
        self.user_tower = UserTower(**user_tower_config)
        self.item_tower = ItemTower(**item_tower_config)

        assert self.user_tower.item_id_emb.num_embeddings == self.item_tower.item_id_emb.num_embeddings
        assert self.user_tower.item_id_emb.embedding_dim == self.item_tower.item_id_emb.embedding_dim

        # Thực hiện Sharing (Tiết kiệm ~6GB VRAM)
        self.user_tower.item_id_emb.weight = self.item_tower.item_id_emb.weight

        if use_learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        assert self.user_tower.output_dim == self.item_tower.output_dim, \
            f"User/Item output dims mismatch: {self.user_tower.output_dim} vs {self.item_tower.output_dim}"

    def forward(self, batch, return_embeddings=False):
        user_emb = self.user_tower(batch)
        item_emb = self.item_tower(batch)
        if return_embeddings:
            return {'user_emb': user_emb, 'item_emb': item_emb}

        if self.temperature.requires_grad: # Chỉ kẹp khi đang học
            with torch.no_grad():
                # Min 0.01: An toàn cho FP16
                # Max 0.5: Tránh mô hình quá lười
                self.temperature.data.clamp_(min=0.02, max=0.5)

        logits = torch.matmul(user_emb, item_emb.T) / self.temperature
        labels = torch.arange(len(user_emb), device=logits.device)
        return {'logits': logits, 'labels': labels, 'temperature': self.temperature.item()}
