import torch
from torch.utils.data import Dataset
import numpy as np
import gc
import psutil
from typing import Dict

class MetadataStore:
    """
    Singleton-like class chịu trách nhiệm quản lý Shared Memory cho Metadata.
    
    Architecture Decision:
    - Sử dụng Pattern Singleton (thông qua `__new__`) để đảm bảo chỉ có duy nhất một bản sao metadata tồn tại trong RAM,
      ngay cả khi khởi tạo nhiều Dataset instance.
    - Dữ liệu được load bằng `mmap_mode='r'` (Memory Mapping), cho phép OS quản lý việc paging in/out,
      giúp tiết kiệm RAM cực lớn khi chạy Multi-process DataLoader (Zero-copy sharing).
    """
    _instance = None

    def __new__(cls, cfg):
        if cls._instance is None:
            print(">>> [MetadataStore] Initializing Shared Static Data...")
            cls._instance = super(MetadataStore, cls).__new__(cls)
            cls._instance.load_data(cfg)
        return cls._instance

    def load_data(self, cfg):
        # 1. Memory-map Embeddings (Read-only, Zero-copy)
        self.embed_matrix = np.load(cfg.EMBEDDINGS_FILE, mmap_mode='r')

        # 2. Load Dense Metadata Arrays
        self.artist_map = np.load(cfg.ARTIST_MAP_FILE)
        self.album_map = np.load(cfg.ALBUM_MAP_FILE)

        self.num_items = len(self.embed_matrix)
        print(f"   ✅ Metadata loaded. Total items: {self.num_items:,}")

class RichFeatureDataset(Dataset):
    """
    Dataset class hiệu năng cao phục vụ Training & Validation.
    
    Features:
    - Virtual Indexing: Ánh xạ index liên tục (0 -> len) sang không gian (User, Time) phức tạp mà không cần tạo list mẫu khổng lồ.
    - On-the-fly Feature Engineering: Tính toán các feature động (Time context, User history stats, Taste drift) ngay tại thời điểm getitem.
    - Memory Efficiency: Sử dụng Memory Map cho toàn bộ dữ liệu lớn, giữ RAM footprint ở mức tối thiểu.
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode

        print(f"\n{'='*60}")
        print(f"Initializing RichFeatureDataset (mode={mode})")
        print(f"{'='*60}")

        # 1. Load Metadata (Singleton)
        print("\n>>> Step 1: Loading MetadataStore...")
        self.meta = MetadataStore(cfg)
        self._print_memory("After MetadataStore")

        # 2. Memory-map flat arrays
        print("\n>>> Step 2: Memory-mapping interactions...")

        self.flat_items = np.memmap(cfg.INTERACTIONS_DIR / "flat_item_ids.npy", dtype='uint32', mode='r')
        self.flat_ts = np.memmap(cfg.INTERACTIONS_DIR / "flat_timestamps.npy", dtype='uint32', mode='r')
        self.offsets = np.load(cfg.INTERACTIONS_DIR / "user_offsets.npy")

        print(f"   Mapped {len(self.flat_items):,} interactions")
        print(f"   Found {len(self.offsets)-1:,} users")
        self._print_memory("After mmap")

        # 3. Build virtual index
        print("\n>>> Step 3: Building virtual index...")
        self._build_virtual_index()
        self._print_memory("After build_index")

        print(f"\n{'='*60}")
        print(f"✅ Dataset ready: {len(self):,} samples for mode '{mode}'")
        print(f"{'='*60}\n")

    def _print_memory(self, label):
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024**3
        print(f"   [{label}] Process RAM: {mem_gb:.2f} GB")

    def _build_virtual_index(self):
        """Build virtual index for train/val/test split"""
        last_ts_chunk = self.flat_ts[-10000:]
        max_ts = np.max(last_ts_chunk)

        test_start_ts = max_ts - (self.cfg.TEST_DAYS * 86400)
        val_start_ts = test_start_ts - (self.cfg.VAL_DAYS * 86400)

        num_users = len(self.offsets) - 1

        sample_users = np.empty(num_users, dtype=np.uint32)
        sample_start_rel = np.empty(num_users, dtype=np.uint32)
        sample_counts = np.empty(num_users, dtype=np.uint32)

        valid_users = 0

        for u in range(num_users):
            start_abs, end_abs = self.offsets[u], self.offsets[u+1]
            u_ts = self.flat_ts[start_abs:end_abs]

            if self.mode == "train":
                stop_idx_rel = np.searchsorted(u_ts, val_start_ts, side='left')
                count = max(0, stop_idx_rel - 1)
                start_rel = 1
            elif self.mode == "val":
                start_idx_rel = np.searchsorted(u_ts, val_start_ts, side='left')
                stop_idx_rel = np.searchsorted(u_ts, test_start_ts, side='left')
                count = max(0, stop_idx_rel - start_idx_rel)
                start_rel = start_idx_rel
            else: # test
                start_idx_rel = np.searchsorted(u_ts, test_start_ts, side='left')
                count = (end_abs - start_abs) - start_idx_rel
                start_rel = start_idx_rel

            if count > 0:
                sample_users[valid_users] = u
                sample_start_rel[valid_users] = start_rel
                sample_counts[valid_users] = count
                valid_users += 1

        self.sample_users = sample_users[:valid_users].copy()
        self.sample_start_rel = sample_start_rel[:valid_users].copy()
        self.sample_counts = sample_counts[:valid_users].copy()

        del sample_users, sample_start_rel, sample_counts
        gc.collect()

        self.cumulative_len = np.zeros(valid_users + 1, dtype=np.int64)
        self.cumulative_len[1:] = np.cumsum(self.sample_counts)

    def __len__(self):
        return int(self.cumulative_len[-1])

    def _compute_drift(self, vec_6m, vec_7d):
        """Compute cosine distance between long-term and short-term taste"""
        norm_6m = np.linalg.norm(vec_6m)
        norm_7d = np.linalg.norm(vec_7d)
        if norm_6m < 1e-6 or norm_7d < 1e-6:
            return np.float32(0.0)
        cos_sim = np.dot(vec_6m, vec_7d) / (norm_6m * norm_7d)
        return np.float32(1.0 - cos_sim)

    def __getitem__(self, idx):
        # 1. Map global index to user
        user_map_idx = np.searchsorted(self.cumulative_len, idx, side='right') - 1
        user_idx_global = int(self.sample_users[user_map_idx])

        # 2. Calculate indices
        user_start_abs = int(self.offsets[user_idx_global])
        start_rel = int(self.sample_start_rel[user_map_idx])
        internal_idx = idx - self.cumulative_len[user_map_idx]

        target_abs_idx = user_start_abs + start_rel + int(internal_idx)

        # 3. Get target info
        target_item_raw = int(self.flat_items[target_abs_idx])
        target_item_shifted = target_item_raw + 1  # +1 for padding
        current_ts = int(self.flat_ts[target_abs_idx])

        # ===================================================================
        # [QUAN TRỌNG] Lấy kích thước thật của embedding từ file (thường là 128)
        # Để tránh lỗi lệch chiều khi tạo vector số 0
        # ===================================================================
        CNN_EMBED_DIM_ACTUAL = self.meta.embed_matrix.shape[1]

        # ===================================================================
        # A. USER FEATURES
        # ===================================================================
        hist_items_raw = self.flat_items[user_start_abs : target_abs_idx]
        hist_ts = self.flat_ts[user_start_abs : target_abs_idx]
        hist_len = len(hist_items_raw)

        # --- Time Features ---
        is_weekend = 1.0 if ((current_ts // 86400) + 4) % 7 >= 5 else 0.0
        hour = (current_ts % 86400) // 3600
        time_slot = 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3))

        # Hour of day (normalized 0-1)
        hour_normalized = float(hour) / 23.0

        # Day of week (normalized 0-1)
        day_of_week = float(((current_ts // 86400) + 4) % 7) / 6.0

        # --- Engagement Bucket ---
        if hist_len < 10:
            eng_bucket = 0
        elif hist_len < 50:
            eng_bucket = 1
        elif hist_len < 200:
            eng_bucket = 2
        else:
            eng_bucket = 3

        # --- Taste Vectors ---
        limit_6m = 500
        start_6m = max(0, hist_len - limit_6m)
        idx_6m = hist_items_raw[start_6m:]

        if len(idx_6m) > 0:
            vecs_6m = self.meta.embed_matrix[idx_6m]
            taste_6m = np.mean(vecs_6m, axis=0).astype(np.float32)
        else:
            # [FIX CRITICAL] Dùng kích thước thật thay vì cfg.EMBED_DIM
            # Nếu cfg.EMBED_DIM = 256 mà file npy = 128 -> Lỗi Stack
            taste_6m = np.zeros(CNN_EMBED_DIM_ACTUAL, dtype=np.float32)

        # Short term (7 days)
        ts_7d_ago = current_ts - (7 * 86400)
        start_7d_rel = np.searchsorted(hist_ts, ts_7d_ago, side='left')
        idx_7d = hist_items_raw[start_7d_rel:]

        if len(idx_7d) > 0:
            vecs_7d = self.meta.embed_matrix[idx_7d]
            taste_7d = np.mean(vecs_7d, axis=0).astype(np.float32)
        else:
            taste_7d = taste_6m  # Fallback

        drift = self._compute_drift(taste_6m, taste_7d)

        # Vector stats 5 chiều
        user_stats_vec = np.array([
            drift,              # [0] Taste drift (0-2)
            is_weekend,         # [1] Weekend flag (0 or 1)
            float(hist_len),    # [2] History length (raw count)
            hour_normalized,    # [3] Hour of day (0-1)
            day_of_week         # [4] Day of week (0-1)
        ], dtype=np.float32)

        # ===================================================================
        # B. SEQUENCE FEATURES (Transformer Input)
        # ===================================================================
        seq_len = self.cfg.MAX_SEQ_LEN
        start_seq = max(0, hist_len - seq_len)

        recent_ids_raw = hist_items_raw[start_seq:]

        # 1. Lấy CNN Audio Embeddings & Ép kiểu FLOAT32 ngay lập tức
        recent_embeds = self.meta.embed_matrix[recent_ids_raw.tolist()].astype(np.float32)

        # 2. Chuẩn bị Item IDs
        recent_ids_shifted = recent_ids_raw.astype(np.int64) + 1

        L = len(recent_ids_raw)
        pad_len = seq_len - L

        if pad_len > 0:
            # Padding Embeddings (Float32) - Dùng kích thước thật
            pad_vec_cnn = np.zeros((pad_len, CNN_EMBED_DIM_ACTUAL), dtype=np.float32)
            seq_embeds_out = np.concatenate([pad_vec_cnn, recent_embeds], axis=0)

            # Padding IDs (Int64)
            pad_vec_ids = np.zeros(pad_len, dtype=np.int64)
            seq_ids_out = np.concatenate([pad_vec_ids, recent_ids_shifted], axis=0)

            # Padding Mask (Bool) - Quan trọng để tránh lỗi Double/Half
            mask_out = np.concatenate([np.ones(pad_len, dtype=np.bool_), np.zeros(L, dtype=np.bool_)])
        else:
            seq_embeds_out = recent_embeds
            seq_ids_out = recent_ids_shifted
            mask_out = np.zeros(seq_len, dtype=np.bool_)


        # ===================================================================
        # C. ITEM & TARGET FEATURES
        # ===================================================================
        item_embed = self.meta.embed_matrix[target_item_raw].astype(np.float32)
        artist_id = int(self.meta.artist_map[target_item_raw])
        album_id = int(self.meta.album_map[target_item_raw])

        return {
            # User features
            "user_taste_6m": torch.from_numpy(taste_6m),
            "user_taste_7d": torch.from_numpy(taste_7d),
            "user_stats_vec": torch.from_numpy(user_stats_vec),
            "user_eng_bucket": torch.tensor(eng_bucket, dtype=torch.long),
            "user_time_slot": torch.tensor(time_slot, dtype=torch.long),

            # Sequence features
            "seq_embeds": torch.from_numpy(seq_embeds_out),
            "seq_item_ids": torch.from_numpy(seq_ids_out),
            "seq_mask": torch.from_numpy(mask_out),

            # Item features
            "item_embed": torch.from_numpy(item_embed),
            "item_artist_id": torch.tensor(artist_id, dtype=torch.long),
            "item_album_id": torch.tensor(album_id, dtype=torch.long),

            # Target
            "target_item_id": torch.tensor(target_item_shifted, dtype=torch.long),
        }

class FastItemInferenceDataset(Dataset):
    """
    Dataset tối giản phục vụ Inference/Indexing tốc độ cao.
    
    Optimization:
    - Bỏ qua toàn bộ logic User/History phức tạp.
    - Chỉ load Item Embeddings và Metadata cần thiết để feed vào Item Tower.
    - Thiết kế để chạy với Batch Size cực lớn (16k - 32k) nhằm tận dụng tối đa GPU throughput.
    """
    def __init__(self, cfg):
        print("⚡ [Dataset] Loading metadata for high-speed inference...")
        self.embed_matrix = np.load(cfg.EMBEDDINGS_FILE, mmap_mode='r')
        self.artist_map = np.load(cfg.ARTIST_MAP_FILE)
        self.album_map = np.load(cfg.ALBUM_MAP_FILE)
        self.num_items = self.embed_matrix.shape[0]
        print(f"   ✅ Loaded {self.num_items:,} items into memory references.")

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        # [FIX CRITICAL] .astype(np.float32)
        # Ép kiểu ngay khi đọc từ đĩa để tránh lỗi Double vs Half
        return (
            self.embed_matrix[idx].astype(np.float32),
            self.artist_map[idx],
            self.album_map[idx],
            idx + 1
        )
