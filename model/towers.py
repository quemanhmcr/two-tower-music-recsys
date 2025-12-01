import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .layers import PositionalEncoding, AttentionPooling

class UserTower(nn.Module):
    """
    User Representation Learning Module.
    
    Architecture:
    - Hybrid Encoder: Kết hợp Transformer (cho Sequential History) và MLP (cho Static Features).
    - Multi-modal Fusion: Tổng hợp thông tin từ Audio Content, User Behavior, và Context (Time/Device).
    
    Input:
    - Sequence Items (IDs + Audio Embeddings)
    - User Stats (Drift, Engagement)
    - Context (Time of day, Day of week)
    
    Output:
    - User Vector (Normalized) nằm trong cùng không gian vector với Item.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dims: list = [512, 256],
        dropout: float = 0.2,
        num_eng_buckets: int = 5,
        num_time_slots: int = 4,
        use_layer_norm: bool = True,
        max_seq_len: int = 50,
        num_items: int = 2_000_001,
        input_audio_dim: int = 128,
        activation: str = "relu"
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # --- 1. Sequence Components ---
        self.item_id_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)

        # Projection Layer
        self.audio_projection = nn.Sequential(
            nn.Linear(input_audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        self.seq_attn_norm = nn.LayerNorm(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_seq_len)
        self.seq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*2,
                dropout=dropout, batch_first=True, norm_first=True
            ),
            num_layers=4
        )
        self.attn_pooling = AttentionPooling(embed_dim)
        self.seq_post_attn_norm = nn.LayerNorm(embed_dim)

        # --- 2. Dense & Categorical Features ---
        self.taste_projection = nn.Sequential(
            nn.Linear(input_audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        self.user_stats_norm = nn.LayerNorm(5)
        self.eng_bucket_emb = nn.Embedding(num_eng_buckets, 16)
        self.eng_bucket_norm = nn.LayerNorm(16)
        self.time_slot_emb = nn.Embedding(num_time_slots, 8)
        self.time_slot_norm = nn.LayerNorm(8)

        # --- 3. MLP Head ---
        input_dim = embed_dim + embed_dim + 5 + 16 + 8 + embed_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # A. Sequence
        seq_ids = batch["seq_item_ids"]
        seq_cnn = batch["seq_embeds"]
        seq_mask = batch["seq_mask"]

        id_embeds = self.item_id_emb(seq_ids)
        cnn_embeds = self.audio_projection(seq_cnn)

        seq_embeds = id_embeds + cnn_embeds
        seq_embeds = self.seq_attn_norm(seq_embeds)
        seq_embeds = self.pos_encoder(seq_embeds)

        seq_encoded = self.seq_encoder(seq_embeds, src_key_padding_mask=seq_mask)
        seq_pooled = self.attn_pooling(seq_encoded, mask=seq_mask)
        seq_pooled = self.seq_post_attn_norm(seq_pooled)

        # B. Other Features
        taste_6m = self.taste_projection(batch["user_taste_6m"])
        taste_7d = self.taste_projection(batch["user_taste_7d"])

        stats = self.user_stats_norm(batch["user_stats_vec"])
        eng_emb = self.eng_bucket_norm(self.eng_bucket_emb(batch["user_eng_bucket"]))
        time_emb = self.time_slot_norm(self.time_slot_emb(batch["user_time_slot"]))

        # C. Concat
        features = torch.cat([
            taste_6m, taste_7d, stats, eng_emb, time_emb, seq_pooled
        ], dim=-1)

        user_emb = self.mlp(features)
        return F.normalize(user_emb, p=2, dim=-1)

class ItemTower(nn.Module):
    """
    Item Representation Learning Module.
    
    Architecture:
    - Content-based: Sử dụng Audio Embeddings (từ CNN pre-trained) làm tín hiệu chính.
    - Collaborative Filtering: Sử dụng ID Embedding để học các đặc trưng ẩn (Latent Factors).
    - Metadata Fusion: Kết hợp thông tin Artist, Album để làm giàu ngữ nghĩa.
    
    Output:
    - Item Vector (Normalized) sẵn sàng cho việc Indexing và Retrieval.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dims: list = [256, 256],
        dropout: float = 0.2,
        num_artists: int = 1_000_000,
        num_albums: int = 2_000_000,
        num_items: int = 2_000_001,
        artist_emb_dim: int = 32,
        album_emb_dim: int = 32,
        use_layer_norm: bool = True,
        input_audio_dim: int = 128,
        activation: str = "relu"
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Embeddings
        self.item_id_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.artist_emb = nn.Embedding(num_artists, artist_emb_dim)
        self.album_emb = nn.Embedding(num_albums, album_emb_dim)

        # Audio Projection
        self.item_audio_projection = nn.Sequential(
            nn.Linear(input_audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        self.item_id_norm = nn.LayerNorm(embed_dim)
        self.artist_norm = nn.LayerNorm(artist_emb_dim)
        self.album_norm = nn.LayerNorm(album_emb_dim)

        # MLP
        input_dim = embed_dim + embed_dim + artist_emb_dim + album_emb_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        dense_vec = self.item_audio_projection(batch["item_embed"])

        id_emb = self.item_id_norm(self.item_id_emb(batch["target_item_id"]))
        artist_emb = self.artist_norm(self.artist_emb(batch["item_artist_id"]))
        album_emb = self.album_norm(self.album_emb(batch["item_album_id"]))

        features = torch.cat([dense_vec, id_emb, artist_emb, album_emb], dim=-1)
        item_emb = self.mlp(features)
        return F.normalize(item_emb, p=2, dim=-1)
