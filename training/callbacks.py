import torch
import os
from pathlib import Path

class CheckpointManager:
    """
    Quản lý việc lưu trữ và xoay vòng (rotation) các Model Checkpoint.
    
    Policy:
    - Top-k Retention: Chỉ giữ lại K checkpoint tốt nhất dựa trên Validation Loss.
    - Auto-cleanup: Tự động xóa các file checkpoint cũ/kém hơn để tiết kiệm ổ cứng.
    - Metadata Saving: Lưu kèm Optimizer state và Epoch info để có thể Resume training bất cứ lúc nào.
    """
    def __init__(self, save_dir='checkpoints', max_to_keep=3):
        self.save_dir = Path(save_dir)
        self.max_to_keep = max_to_keep
        self.best_checkpoints = []
        self.save_dir.mkdir(exist_ok=True)

    def save(self, model, optimizer, epoch, val_loss, extra_info=None):
        should_save = len(self.best_checkpoints) < self.max_to_keep or val_loss < self.best_checkpoints[-1]['loss']
        if not should_save: return

        filename = f"model_ep{epoch:03d}_loss{val_loss:.4f}.pt"
        filepath = self.save_dir / filename

        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'extra_info': extra_info
        }
        torch.save(save_dict, filepath)
        print(f"✅ Saved Top-{self.max_to_keep} Checkpoint: {filename}")

        self.best_checkpoints.append({'path': filepath, 'loss': val_loss})
        self.best_checkpoints.sort(key=lambda x: x['loss'])

        if len(self.best_checkpoints) > self.max_to_keep:
            to_remove = self.best_checkpoints.pop(-1)
            try:
                os.remove(to_remove['path'])
                print(f"🗑️ Removed old checkpoint: {to_remove['path'].name}")
            except OSError as e:
                print(f"⚠️ Error removing file: {e}")
