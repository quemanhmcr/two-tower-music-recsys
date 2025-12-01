import faiss
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

def test_retrieval(trainer, config, top_k=10):
    """
    Sanity Check: Kiểm tra chất lượng mô hình bằng mắt thường (Qualitative Evaluation).
    
    Logic:
    1. Lấy ngẫu nhiên 1 User từ tập Validation.
    2. Sinh User Vector bằng User Tower.
    3. Truy vấn (Query) vào FAISS Index để tìm Top-K Item tương đồng nhất.
    4. Hiển thị kết quả kèm trạng thái (Re-listen hay New Discovery) để đánh giá sơ bộ độ hợp lý.
    """
    print("🔍 STARTING RETRIEVAL TEST...")

    # 1. Load FAISS Index
    index_path = f"{config.CHECKPOINT_DIR}/item_vectors_256d.faiss"
    if not os.path.exists(index_path):
        print(f"❌ Error: File index không tồn tại tại {index_path}")
        return

    print(f"   📂 Loading Index from: {index_path}")
    try:
        index = faiss.read_index(index_path)
        print(f"   ✅ Index loaded. Total Items: {index.ntotal:,}")
    except Exception as e:
        print(f"❌ Lỗi load index: {e}")
        return

    # 2. Lấy 1 Batch User (FIX LỖI ZOMBIE PROCESS)
    print("\n👤 Generating User Vector...")
    try:
        # [FIX] Tạo loader tạm với num_workers=0 để tránh lỗi multiprocessing cleanup
        # Ta tái sử dụng dataset của val_loader cũ
        temp_loader = DataLoader(
            trainer.val_loader.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0 # <--- QUAN TRỌNG: Chạy trên main process
        )
        batch = next(iter(temp_loader))

        device = trainer.device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Chạy User Tower
        user_tower = trainer.model.user_tower
        user_tower.eval()

        with torch.no_grad():
            user_embeddings = user_tower(batch)
            query_vector = user_embeddings[0].cpu().numpy().reshape(1, -1)

    except Exception as e:
        print(f"❌ Lỗi sinh user vector: {e}")
        return

    # 3. Search & Print Results
    print(f"\n🔎 Searching Top-{top_k} recommendations...")
    D, I = index.search(query_vector, top_k)

    print("\n" + "="*50)
    print("🎯 RECOMMENDATION RESULTS")
    print("="*50)

    history_ids = batch['seq_item_ids'][0].cpu().numpy()
    history_ids = history_ids[history_ids > 0] # Bỏ padding

    print(f"User History (Last 5): {history_ids[-5:]}")
    print("-" * 65)
    print(f"{'Rank':<5} | {'Item ID':<12} | {'Score':<10} | {'Status'}")
    print("-" * 65)

    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        real_item_id = idx + 1
        status = "🎧 Re-listen" if real_item_id in history_ids else "✨ New"
        print(f"{rank+1:<5} | {real_item_id:<12} | {score:.4f}     | {status}")

    print("="*50)
