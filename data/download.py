from huggingface_hub import hf_hub_download
import polars as pl
from pathlib import Path
from config import TrainingConfig

def download_data():
    """
    Module chịu trách nhiệm tải dữ liệu thô từ HuggingFace Hub.
    
    Workflow:
    1. Tải file `listens.parquet` (hành vi người dùng).
    2. Tải các file metadata (embeddings, artist/album mapping).
    3. Lưu trữ vào thư mục local được định nghĩa trong TrainingConfig.
    
    Note:
    - Sử dụng `hf_hub_download` để đảm bảo tính toàn vẹn của file (checksum verification).
    - Dữ liệu được tải về dạng Parquet để tối ưu I/O speed khi đọc bằng Polars.
    """
    
    # Init config để lấy đường dẫn chuẩn
    cfg = TrainingConfig()
    
    # Tạo thư mục tạm và thư mục đích
    temp_dir = cfg.DATA_ROOT / "temp_data"
    output_dir = cfg.DATA_ROOT / "yambda_data"
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("⬇️ Downloading file directly...")
    local_file_path = hf_hub_download(
        repo_id="yandex/yambda",
        filename="flat/500m/listens.parquet",
        repo_type="dataset",
        local_dir=temp_dir 
    )
    print(f"✅ File downloaded to: {local_file_path}")

    print("🎵 LOADING MUSIC-RELATED DATA")
    print("=" * 80)

    # Files to download
    files = [
        ("embeddings.parquet", "audio embeddings"),
        ("artist_item_mapping.parquet", "artist-item mapping"),
        ("album_item_mapping.parquet", "album-item mapping"),
    ]

    dataframes = {}

    for filename, description in files:
        print(f"\n📥 Downloading {description}...")

        # Download trực tiếp từ HF Hub → local path
        local_path = hf_hub_download(
            repo_id="yandex/yambda",
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # Copy thật, không symlink
        )

        # Load với Polars để verify + get row count
        df = pl.scan_parquet(local_path)
        row_count = df.select(pl.len()).collect().item()

        print(f"✅ {description}: {row_count:,} rows")
        print(f"   📁 Saved to: {local_path}")

        # Store reference
        dataframes[filename.replace(".parquet", "")] = local_path

    print("\n" + "=" * 80)
    print("✅ All files downloaded to:", output_dir)
    print("\nFile paths:")
    for name, path in dataframes.items():
        print(f"   {name}: {path}")

if __name__ == "__main__":
    download_data()
