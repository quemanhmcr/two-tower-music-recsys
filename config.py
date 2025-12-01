from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class TrainingConfig:
    """
    Configuration Class quản lý toàn bộ hyperparameter và đường dẫn của dự án.
    
    Design Pattern: Configuration Object
    Mục đích: Tập trung hóa cấu hình, giúp dễ dàng quản lý experiment và reproducibility.
    
    Attributes:
        DATA_ROOT (Path): Đường dẫn gốc chứa dữ liệu. Mặc định trỏ về folder `data/` tại root project.
    """
    # --- Paths ---
    # Chuyển về đường dẫn relative "data" tính từ root project (nơi chạy script)
    # Senior Tip: Luôn dùng Pathlib thay vì string path để đảm bảo OS-agnostic (chạy tốt trên cả Win/Linux/Mac)
    DATA_ROOT: Path = Path("data")

    # Static data
    STATIC_DIR: Path = DATA_ROOT / "static"
    EMBEDDINGS_FILE: Path = STATIC_DIR / "embeddings.npy"
    ARTIST_MAP_FILE: Path = STATIC_DIR / "artists.npy"
    ALBUM_MAP_FILE: Path = STATIC_DIR / "albums.npy"

    # Interaction data
    INTERACTIONS_DIR: Path = DATA_ROOT / "interactions"

    # --- Feature Engineering ---
    MAX_SEQ_LEN: int = 50
    EMBED_DIM: int = 256
    HIDDEN_DIMS_USER: List[int] = field(default_factory=lambda: [512, 256])
    HIDDEN_DIMS_ITEM: List[int] = field(default_factory=lambda: [256, 256])

    # --- Data Split ---
    VAL_DAYS: int = 7
    TEST_DAYS: int = 7

    # --- Training Hyperparameters ---
    BATCH_SIZE: int = 2048
    ACCMULATION_STEPS: int = 2
    NUM_EPOCHS: int = 1
    NUM_WORKERS: int = 4 
    LR: float = 2e-4
    WEIGHT_DECAY: float = 0.02
    WARMUP_RATIO: float = 0.05
    TEMPERRATURE: float = 0.1
    VAL_INTERVAL: int = 5000
    VAL_BATCHES_LIMIT: int = 500

    # --- Checkpointing ---
    # Lưu checkpoint ngay trong folder data để dễ quản lý
    CHECKPOINT_DIR: str = str(DATA_ROOT / 'checkpoints')
    MAX_CHECKPOINTS: int = 3

    USE_LEARNABLE: bool = True
