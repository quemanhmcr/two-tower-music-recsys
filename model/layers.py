import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Inject thông tin vị trí vào sequence embedding (Standard Transformer PE).
    
    Math:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Mục đích: Giúp model phân biệt được thứ tự của các item trong lịch sử nghe nhạc (gần đây vs lâu rồi).
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AttentionPooling(nn.Module):
    """
    Cơ chế Pooling thông minh dựa trên Attention Weight.
    
    Thay vì Average Pooling (lấy trung bình cộng) hay Max Pooling, Attention Pooling cho phép model
    tự học cách "trọng số hóa" tầm quan trọng của từng item trong sequence.
    
    Ví dụ: Một bài hát nghe 100 lần sẽ có weight cao hơn bài nghe lướt qua.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1, bias=False)
        )

    def forward(self, x, mask=None):
        scores = self.attention(x)
        if mask is not None:
            # [FIX CRITICAL]: Đổi -1e9 thành -1e4 để tránh lỗi tràn số Float16 (Half)
            # -1e4 là đủ nhỏ để Softmax về 0, và nằm trong vùng an toàn của Float16
            scores = scores.masked_fill(mask.unsqueeze(-1), -1e4)
        weights = F.softmax(scores, dim=1)
        output = torch.sum(x * weights, dim=1)
        return output
