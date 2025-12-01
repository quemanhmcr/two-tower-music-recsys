import faiss
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import gc
from ..data.dataset import FastItemInferenceDataset

def optimized_collate_fn(batch):
    """Custom collate function"""
    embeds, artists, albums, ids = zip(*batch)
    return {
        # [FIX CRITICAL] Đảm bảo Tensor là Float32
        "item_embed": torch.from_numpy(np.stack(embeds)).to(torch.float32),
        "item_artist_id": torch.tensor(artists, dtype=torch.long),
        "item_album_id": torch.tensor(albums, dtype=torch.long),
        "target_item_id": torch.tensor(ids, dtype=torch.long)
    }

def generate_and_index(trainer, config):
    """
    Quy trình Inference & Indexing tổng thể.
    
    Workflow:
    1. Embedding Generation: Chạy Item Tower (Forward Pass) cho toàn bộ tập Item để sinh ra vector đại diện.
    2. FAISS Indexing: Đưa các vector này vào cấu trúc dữ liệu FAISS (Facebook AI Similarity Search) để phục vụ tìm kiếm vector tốc độ cao.
    3. Serialization: Lưu Index xuống đĩa để dùng cho Serving API.
    
    Performance Note:
    - Sử dụng `torch.inference_mode()` và `autocast` để tối đa hóa tốc độ sinh vector.
    - Batch Size lớn (16k) giúp bão hòa GPU Core.
    """
    print(f"\n🚀 STARTING HIGH-PERFORMANCE INFERENCE...")

    # Clean Memory
    if hasattr(trainer, 'optimizer'): del trainer.optimizer
    if hasattr(trainer, 'scaler'): del trainer.scaler
    gc.collect()
    torch.cuda.empty_cache()

    # Config
    BATCH_SIZE = 16384
    NUM_WORKERS = 2

    dataset = FastItemInferenceDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=optimized_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Model & Device Setup
    model = trainer.model.item_tower
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-move model to GPU
    if next(model.parameters()).device.type != 'cuda':
        print(f"⚠️ Model is on CPU. Moving to {device}...")
        model = model.to(device)
    else:
        print(f"✅ Model is already on {device}.")

    model.eval()

    # Output Buffer
    final_embeddings = np.zeros((len(dataset), config.EMBED_DIM), dtype=np.float32)
    print(f"   🔥 Processing with Batch Size: {BATCH_SIZE}")

    current_idx = 0

    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            for batch in tqdm(dataloader, desc="⚡ Inferencing"):
                # Move to GPU
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                # Forward Pass
                outputs = model(batch)

                # Copy results
                batch_len = outputs.shape[0]
                final_embeddings[current_idx : current_idx + batch_len] = outputs.cpu().numpy()
                current_idx += batch_len

    print(f"✅ Inference Complete. Shape: {final_embeddings.shape}")

    # Build FAISS
    print(f"\n🏗️ BUILDING FAISS INDEX (Exact Search)...")
    index = faiss.IndexFlatIP(config.EMBED_DIM)
    index.add(final_embeddings)

    save_path = f"{config.CHECKPOINT_DIR}/item_vectors_256d.faiss"
    faiss.write_index(index, save_path)

    print(f"🎉 SUCCESS! Saved to: {save_path}")
    return index, final_embeddings
