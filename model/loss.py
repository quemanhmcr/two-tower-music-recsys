import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Information Noise Contrastive Estimation).
    
    Standard Loss cho bài toán Contrastive Learning / Retrieval.
    - Positive Sample: Cặp (User, Item) thực tế trong batch.
    - Negative Samples: Các Item còn lại trong batch (In-batch Negatives).
    
    Mục đích: Kéo vector User và Positive Item lại gần nhau, đẩy xa các Negative Items.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, logits, labels):
        return self.criterion(logits, labels)

def ranking_loss_fn(pred_ratio, target_ratio):
    """
    Loss Function cho bài toán Ranking (Regression/Classification).
    
    Strategy:
    - Sử dụng Binary Cross Entropy (BCE) thay vì MSE ngay cả khi target là số thực (0.0 -> 1.0).
    - Lý do: BCE phạt nặng hơn khi dự đoán sai ở 2 cực (0 và 1), phù hợp với bản chất xác suất của bài toán CTR.
    """
    return F.binary_cross_entropy(pred_ratio, target_ratio)
