import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLayer(nn.Module):
    """
    Explicit Feature Interaction Layer (DCN-v2).
    
    Công thức: x_{l+1} = x_0 * (x_l * w) + b + x_l
    
    Ý nghĩa:
    - Học các tương tác bậc cao (High-order interactions) một cách tường minh.
    - Giúp model trả lời câu hỏi: "Nếu User thích A VÀ Item là B thì xác suất click là bao nhiêu?"
    - Hiệu quả hơn MLP truyền thống trong việc bắt các pattern tổ hợp (Combinatorial Patterns).
    """
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, 1)) # Vector cột
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x0, xi):
        # x0: input gốc, xi: input lớp trước
        # Tính feature crossing
        interaction = torch.matmul(x0.unsqueeze(-1), xi.unsqueeze(1)) # Outer product ảo
        # Thực tế cài đặt tối ưu: (xi . w) * x0
        inner = torch.matmul(xi, self.weight) # (Batch, 1)
        return x0 * inner + self.bias + xi

class RankingModel(nn.Module):
    """
    Stage 2: Ranking / Re-ranking Model.
    
    Architecture: Deep & Cross Network (DCN-v2).
    
    Workflow:
    1. Nhận input là Embedding từ User Tower và Item Tower (đã pre-trained & frozen).
    2. Feature Fusion: Ghép nối (Concat) 2 vector lại.
    3. Cross Network: Học tương tác tường minh.
    4. Deep Network: Học tương tác phi tuyến tính.
    5. Output: Xác suất (Probability) User sẽ tương tác với Item (CTR/Conversion Rate).
    """
    def __init__(self, two_tower_model, feature_dim=512):
        super().__init__()
        # --------------------------------------------------------
        # A. PRE-TRAINED COMPONENTS (Feature Extractors)
        # --------------------------------------------------------
        # Load lại User/Item Tower đã train xong
        # Freeze để tiết kiệm VRAM và giữ tính chất ngữ nghĩa đã học tốt
        self.user_tower = two_tower_model.user_tower
        self.item_tower = two_tower_model.item_tower

        for param in self.user_tower.parameters(): param.requires_grad = False
        for param in self.item_tower.parameters(): param.requires_grad = False

        # Input dimension sau khi concat (User Emb + Item Emb + Extra Features)
        self.input_dim = self.user_tower.output_dim + self.item_tower.output_dim # 256 + 256 = 512

        # --------------------------------------------------------
        # B. DCN-V2 STRUCTURE (The Ranking Brain)
        # --------------------------------------------------------
        # 1. Cross Network: Học tương tác feature (Explicit)
        self.num_cross_layers = 3
        self.cross_layers = nn.ModuleList([
            CrossLayer(self.input_dim) for _ in range(self.num_cross_layers)
        ])

        # 2. Deep Network: Học phi tuyến tính (Implicit)
        self.deep_mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 3. Final Prediction Head
        # Kết hợp output của Cross Net và Deep Net
        self.final_head = nn.Linear(self.input_dim + 64, 1)

    def forward(self, batch):
        # 1. Extract Embeddings (Dùng Pre-trained Towers)
        with torch.no_grad(): # Không tính grad cho tower cũ
            user_emb = self.user_tower(batch) # [Batch, 256]
            item_emb = self.item_tower(batch) # [Batch, 256]

        # 2. Feature Fusion
        # Tại đây có thể concat thêm feature tương tác khác (vd: user_age, item_genre)
        x0 = torch.cat([user_emb, item_emb], dim=-1) # [Batch, 512]

        # 3. DCN Forward
        # - Deep Path
        x_deep = self.deep_mlp(x0)

        # - Cross Path
        x_cross = x0
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)

        # 4. Stack & Predict
        x_final = torch.cat([x_cross, x_deep], dim=-1)
        logits = self.final_head(x_final)

        # Output sigmoid để ép về khoảng [0, 1] (played_ratio)
        return torch.sigmoid(logits).squeeze()
