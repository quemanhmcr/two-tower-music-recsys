import torch
import time
from collections import defaultdict
from model.loss import InfoNCELoss
from .callbacks import CheckpointManager

class TwoTowerTrainer:
    """
    Trainer Class quản lý toàn bộ vòng đời Training (Loop, Validation, Logging).
    
    Key Optimizations:
    - Mixed Precision Training (AMP): Tự động ép kiểu Float16 cho phép tính nặng, giữ Float32 cho việc tích lũy gradient -> Tăng tốc 2x, giảm 50% VRAM.
    - Gradient Accumulation: Giả lập Batch Size lớn bằng cách tích lũy gradient qua nhiều bước nhỏ -> Ổn định training.
    - Memory-efficient Validation: Validation loop được thiết kế để không lưu trữ Logits/Tensors thừa, tránh OOM (Out Of Memory).
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device='cuda',
        use_amp=True,
        gradient_clip=1.0,
        log_interval=100,
        accumulation_steps=2,      # [CONFIG] Tích lũy gradient (mặc định 2)
        val_interval=5000,         # [NEW] Validate sau mỗi 5000 bước
        val_batches_limit=500      # [NEW] Chỉ validate 500 batch cho nhanh
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval

        # Cấu hình mới
        self.accumulation_steps = accumulation_steps
        self.val_interval = val_interval
        self.val_batches_limit = val_batches_limit

        self.criterion = InfoNCELoss()
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        self.global_step = 0

        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []

        # Placeholder Checkpoint
        self.checkpoint_manager = CheckpointManager(save_dir='checkpoints', max_to_keep=3)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # 1. Move data
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 2. Forward & Loss
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch)
                    loss = self.criterion(outputs['logits'], outputs['labels'])
                    loss = loss / self.accumulation_steps # Chia nhỏ loss
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(batch)
                loss = self.criterion(outputs['logits'], outputs['labels'])
                loss = loss / self.accumulation_steps
                loss.backward()

            current_loss_val = loss.item() * self.accumulation_steps
            total_loss += current_loss_val

            # 3. Optimize (Step)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                if self.scheduler: self.scheduler.step()
                self.global_step += 1

            # 4. Logging
            if batch_idx % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {current_loss_val:.4f} | Temp: {outputs['temperature']:.4f} | "
                      f"LR: {lr:.6f} | Time: {elapsed:.1f}s")
                start_time = time.time()

            # 5. [NEW] Mid-Epoch Validation (Validate thường xuyên)
            if (batch_idx + 1) % self.val_interval == 0:
                print(f"\n>>> [Mid-Epoch Val] Step {batch_idx+1}: Running quick validation ({self.val_batches_limit} batches)...")
                val_loss, val_metrics = self.validate(max_batches=self.val_batches_limit)

                print(f"   >> Val Loss: {val_loss:.4f}")
                print(f"   >> Metrics: {val_metrics}")
                print(f"{'-'*60}")

                # Lưu history để vẽ biểu đồ sau này
                self.val_losses.append(val_loss)
                self.val_metrics_history.append(val_metrics)

                # QUAN TRỌNG: Phải chuyển lại về mode Train sau khi val
                self.model.train()

        return total_loss / len(self.train_loader)

    # Định nghĩa lại hàm validate tối ưu bộ nhớ
    @torch.no_grad()
    def validate(self, max_batches=None):
        self.model.eval()
        total_loss = 0

        # Dùng dictionary để lưu tổng metrics
        total_metrics = defaultdict(float)

        limit = self.val_batches_limit if max_batches is None else max_batches
        if limit == -1: limit = float('inf')

        val_steps = 0

        print(f"Running optimized validation on {limit if limit != float('inf') else 'all'} batches...")

        for i, batch in enumerate(self.val_loader):
            # 1. Move to GPU
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 2. Forward (Tốn VRAM ở đây, nhưng sẽ được giải phóng ngay sau vòng lặp)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch)
                    loss = self.criterion(outputs['logits'], outputs['labels'])
            else:
                outputs = self.model(batch)
                loss = self.criterion(outputs['logits'], outputs['labels'])

            total_loss += loss.item()

            # 3. TÍNH METRIC NGAY LẬP TỨC (Không lưu logits)
            # batch_metrics trả về số scalar (float), không tốn VRAM
            batch_metrics = self.compute_metrics(outputs['logits'], outputs['labels'])

            for k, v in batch_metrics.items():
                total_metrics[k] += v

            # 4. Dọn dẹp thủ công (Optional nhưng tốt)
            del outputs, loss

            val_steps += 1
            if val_steps >= limit: break

        # Tính trung bình
        avg_loss = total_loss / val_steps if val_steps > 0 else 0
        avg_metrics = {k: v / val_steps for k, v in total_metrics.items()}

        # Dọn dẹp cache GPU sau khi val xong để trả lại chỗ cho Training
        torch.cuda.empty_cache()

        return avg_loss, avg_metrics

    def compute_metrics(self, logits, labels):
        metrics = {}
        # Move to CPU to save GPU memory during calculation (Optional)
        # logits, labels = logits.cpu(), labels.cpu()

        for k in [1, 5, 10]:
            if k >= logits.size(1): continue
            _, topk_indices = torch.topk(logits, k, dim=1)
            hits = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()
            metrics[f'R@{k}'] = hits.mean().item()

        sorted_indices = torch.argsort(logits, dim=1, descending=True)
        ranks = (sorted_indices == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
        mrr = (1.0 / ranks.float()).mean().item()
        metrics['MRR'] = mrr
        return metrics

    def fit(self, num_epochs: int):
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Settings: Accumulation={self.accumulation_steps}, Val_Interval={self.val_interval} steps")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}")

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            print("\n>>> End of Epoch Validation (Running Full Validation)...")
            # Cuối epoch thì chạy val nhiều hơn (hoặc full) để chính xác
            val_loss, val_metrics = self.validate(max_batches=2000)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Metrics: {val_metrics}")

            self.checkpoint_manager.save(self.model, self.optimizer, epoch, val_loss, val_metrics)
