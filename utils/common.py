import psutil
import numpy as np
from pathlib import Path
from typing import Tuple

def print_memory_usage(label=""):
    """
    Utility: Theo dõi mức tiêu thụ RAM của Process hiện tại và toàn hệ thống.
    Rất hữu ích để debug Memory Leak trong các pipeline xử lý dữ liệu lớn.
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3

    # System memory
    vm = psutil.virtual_memory()
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"Process RAM: {mem_gb:.2f} GB")
    print(f"System Total: {vm.total / 1024**3:.2f} GB")
    print(f"System Available: {vm.available / 1024**3:.2f} GB")
    print(f"System Used: {vm.percent}%")
    print(f"{'='*60}\n")

def get_vocab_sizes_from_npy(
    artist_map_file: Path,
    album_map_file: Path,
    embeddings_file: Path,
    add_padding: bool = True
) -> Tuple[int, int, int]:
    """
    Tính toán kích thước Vocabulary (số lượng Artist, Album, Item) trực tiếp từ file dữ liệu đã xử lý.
    Đảm bảo Model Config luôn khớp 100% với dữ liệu thực tế, tránh lỗi Dimension Mismatch.
    """
    print(f"\n{'='*40}\n🚀 CALCULATING VOCAB SIZES (FROM .NPY FILES)\n{'='*40}")

    def get_size(file_path: Path, label: str, take_max: bool = True):
        if not file_path.exists():
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        # Load mmap_mode để không tốn RAM
        arr = np.load(file_path, mmap_mode='r')

        # Với artist/album, max_id là vocab size
        if take_max:
             # +1 vì ID bắt đầu từ 0
            size = np.max(arr) + 1
        # Với item (embeddings), số dòng là vocab size
        else:
            size = len(arr)

        # +1 cho padding token
        if add_padding:
            size += 1

        print(f"✅ {label}:")
        print(f"   ├─ File: {file_path.name}")
        print(f"   └─ Final Vocab Size: {size:,} (Padding={'Yes' if add_padding else 'No'})")
        return int(size)

    # Artist & Album: vocab size = max_id + 1
    num_artists = get_size(artist_map_file, "Artists", take_max=True)
    num_albums = get_size(album_map_file, "Albums", take_max=True)

    # Items: vocab size = số lượng embedding vectors
    num_items = get_size(embeddings_file, "Items (Tracks)", take_max=False)

    print(f"{'-'*40}\n🎯 CONFIG OUTPUT:")
    print(f"num_items={num_items}, num_artists={num_artists}, num_albums={num_albums}\n{'='*40}")

    return num_artists, num_albums, num_items
