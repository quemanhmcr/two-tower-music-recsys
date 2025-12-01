import polars as pl
import numpy as np
from pathlib import Path
import gc
import shutil
from tqdm import tqdm
from config import TrainingConfig

def process_data():
    """
    Pipeline xử lý dữ liệu thô: Remapping ID & Cleaning.
    
    Logic chính:
    1. Intersection: Chỉ giữ lại các Item ID xuất hiện trong cả file Listens (hành vi) và Embeddings (content).
    2. Remapping: Ánh xạ Item ID gốc (string/int lộn xộn) sang ID liên tục (0 -> N-1) để tối ưu cho Embedding Layer.
    3. Consistency: Đảm bảo Metadata (Artist, Album) cũng được re-map theo ID mới.
    
    Output:
    - Các file Parquet đã được làm sạch và đồng bộ ID.
    - Sẵn sàng cho bước tạo Static Features.
    """
    
    # ==============================================================================
    # CONFIGURATION
    # ==============================================================================
    cfg = TrainingConfig()
    
    OLD_DATA_DIR = cfg.DATA_ROOT / "temp_data"
    META_DATA_DIR = cfg.DATA_ROOT / "yambda_data"

    INPUT_LISTENS = OLD_DATA_DIR / "listens.parquet"
    INPUT_EMBEDDINGS = META_DATA_DIR / "embeddings.parquet"
    INPUT_ARTIST_MAP = META_DATA_DIR / "artist_item_mapping.parquet"
    INPUT_ALBUM_MAP = META_DATA_DIR / "album_item_mapping.parquet"

    OUTPUT_DIR = cfg.DATA_ROOT / "remapped_data"
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 STARTING ID RE-MAPPING PIPELINE (INTERSECTION LOGIC)")
    print("="*60)

    # ==============================================================================
    # STEP 1: TẠO MAPPING TỪ GIAO ĐIỂM (CRITICAL STEP)
    # ==============================================================================
    print("\n[1/5] ⚔️  Finding Intersection (Listens ∩ Embeddings)...")

    # 1. Quét Unique IDs từ Listens
    #    (Những bài người dùng đã nghe)
    q_listens_ids = pl.scan_parquet(INPUT_LISTENS).select("item_id").unique()

    # 2. Quét Unique IDs từ Embeddings
    #    (Những bài hệ thống có vector)
    #    Lưu ý: Cần detect tên cột embedding là 'item_id' hay tên khác để join cho đúng
    q_embed_ids = pl.scan_parquet(INPUT_EMBEDDINGS).select("item_id").unique()

    # 3. Lấy GIAO ĐIỂM (INNER JOIN)
    #    Chỉ giữ lại những bài VỪA được nghe VỪA có vector
    valid_ids = (
        q_listens_ids.join(q_embed_ids, on="item_id", how="inner")
        .collect()  # Thực thi để lấy danh sách sạch về RAM
    )

    total_valid = len(valid_ids)
    print(f"   --> Found {total_valid:,} valid items (Intersection).")
    print(f"       (Items without embeddings will be dropped automatically)")

    # 4. Tạo Map chuẩn (0 -> N-1)
    id_map = valid_ids.sort("item_id").with_columns(
        pl.arange(0, pl.len(), dtype=pl.UInt32).alias("new_id")
    )

    ID_MAP_FILE = OUTPUT_DIR / "id_mapping.parquet"
    id_map.write_parquet(ID_MAP_FILE)
    print(f"   --> Saved clean mapping to: {ID_MAP_FILE}")

    # Dọn dẹp
    del valid_ids, q_listens_ids, q_embed_ids
    gc.collect()

    # ==============================================================================
    # STEP 2: RE-MAP LISTENS (Sẽ tự động lọc bỏ bài ko có embedding)
    # ==============================================================================
    print("\n[2/5] 🎧 Re-mapping 'listens.parquet'...")

    lf_listens = pl.scan_parquet(INPUT_LISTENS)
    lf_map = pl.scan_parquet(ID_MAP_FILE)

    # Inner Join ở đây sẽ LOẠI BỎ những dòng nghe nhạc mà bài hát đó không có trong Map
    # (tức là bài hát ko có embedding)
    lf_new_listens = (
        lf_listens.join(lf_map, on="item_id", how="inner")
        .drop("item_id")
        .rename({"new_id": "item_id"})
    )

    NEW_LISTENS_FILE = OUTPUT_DIR / "listens.parquet"
    lf_new_listens.sink_parquet(NEW_LISTENS_FILE)
    print(f"   --> ✅ Saved filtered listens to {NEW_LISTENS_FILE}")
    gc.collect()

    # ==============================================================================
    # STEP 3: RE-MAP METADATA
    # ==============================================================================
    print("\n[3/5] 📚 Re-mapping Metadata...")

    def remap_metadata(input_path, output_name):
        if not input_path.exists(): return

        print(f"   ... Processing {input_path.name}")
        lf_meta = pl.scan_parquet(input_path)
        lf_map = pl.scan_parquet(ID_MAP_FILE)

        lf_new_meta = (
            lf_meta.join(lf_map, on="item_id", how="inner")
            .drop("item_id")
            .rename({"new_id": "item_id"})
        )
        lf_new_meta.sink_parquet(OUTPUT_DIR / output_name)

    remap_metadata(INPUT_ARTIST_MAP, "artist_item_mapping.parquet")
    remap_metadata(INPUT_ALBUM_MAP, "album_item_mapping.parquet")
    gc.collect()

    # ==============================================================================
    # STEP 4: RE-MAP EMBEDDINGS (Đảm bảo khớp 100%)
    # ==============================================================================
    print("\n[4/5] 🧬 Re-mapping CNN Embeddings...")

    if INPUT_EMBEDDINGS.exists():
        schema = pl.scan_parquet(INPUT_EMBEDDINGS).schema
        emb_col = "embed" if "embed" in schema else "embedding"

        lf_emb = pl.scan_parquet(INPUT_EMBEDDINGS)
        lf_map = pl.scan_parquet(ID_MAP_FILE)

        # Join với Map (vốn được tạo từ chính embedding) -> Chắc chắn giữ lại đúng số lượng
        lf_new_emb = (
            lf_emb.join(lf_map, on="item_id", how="inner")
            .select(["new_id", emb_col])
            .rename({emb_col: "embedding"})
            .sort("new_id") # Quan trọng: Xếp đúng thứ tự 0, 1, 2...
        )

        NEW_EMB_FILE = OUTPUT_DIR / "embeddings.parquet"
        lf_new_emb.sink_parquet(NEW_EMB_FILE)

        # Convert to Numpy
        print("   ... Creating Numpy Matrix...")
        df_final = pl.read_parquet(NEW_EMB_FILE)
        matrix = np.stack(df_final["embedding"].to_numpy())

        NEW_EMB_NPY = OUTPUT_DIR / "embeddings_mmap.npy"
        np.save(NEW_EMB_NPY, matrix)

        print(f"   --> ✅ Matrix Shape: {matrix.shape}")

        # Validation logic
        if matrix.shape[0] == total_valid:
            print("   --> ✅ Integrity Check: Embeddings count matches Map count exactly.")
        else:
            print(f"   --> ❌ ERROR: Shape mismatch ({matrix.shape[0]} vs {total_valid})")

        del matrix, df_final
    else:
        print("   ❌ Error: Embeddings file missing.")

    gc.collect()

    # ==============================================================================
    # STEP 5: FINAL CHECK
    # ==============================================================================
    print("\n[5/5] ✅ FINAL VERIFICATION")
    # Check file listens mới
    df_check = pl.read_parquet(NEW_LISTENS_FILE)
    listens_unique = df_check["item_id"].n_unique()
    max_id = df_check["item_id"].max()

    print(f"Expected Unique : {total_valid:,}")
    print(f"Listens Unique  : {listens_unique:,}")
    print(f"Max ID          : {max_id:,}")

    if listens_unique == total_valid and max_id == (total_valid - 1):
        print("\n🎉 SUPER SUCCESS: Listens and Embeddings are now PERFECTLY synced.")
        print(f"   Items processed: {total_valid:,}")
        print(f"   Ready for training at: {OUTPUT_DIR}")
    else:
        print("\n❌ Still inconsistent. Check logs.")

def build_static_data():
    """
    Chuyển đổi dữ liệu tĩnh (Embeddings, Artists, Albums) sang định dạng Numpy Dense.
    
    Optimization:
    - Sử dụng `numpy.save` để lưu binary file, cho phép load cực nhanh bằng `mmap_mode`.
    - Chuyển đổi Sparse Mapping (Parquet) sang Dense Array (Numpy) để truy xuất O(1) theo Item ID.
    """
    
    cfg = TrainingConfig()
    
    # CONFIG
    INPUT_DIR = cfg.DATA_ROOT / "remapped_data"  # Nơi chứa file parquet đã map ID
    OUTPUT_DIR = cfg.STATIC_DIR # Nơi lưu npy

    # INPUT FILES
    FILE_EMBED = INPUT_DIR / "embeddings.parquet"
    FILE_ARTIST = INPUT_DIR / "artist_item_mapping.parquet"
    FILE_ALBUM = INPUT_DIR / "album_item_mapping.parquet"

    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 BUILDING STATIC FEATURES (Embeddings & Metadata)")
    print(f"   Input: {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print("="*60)

    # ---------------------------------------------------------
    # 1. EMBEDDINGS (Parquet -> Numpy Mmap)
    # ---------------------------------------------------------
    print("\n[1/3] 🧬 Processing Embeddings...")

    # Scan để lấy shape
    df_emb_schema = pl.scan_parquet(FILE_EMBED)
    max_id = df_emb_schema.select(pl.col("new_id").max()).collect().item()
    N_ITEMS = max_id + 1
    EMBED_DIM = 128 # Giả định, hoặc lấy len của vector đầu tiên

    print(f"   Total Items: {N_ITEMS:,}")

    # Load toàn bộ vào RAM (với 2.5M items * 128 float32 ~ 1.2GB RAM -> Khả thi trên Colab)
    # Nếu RAM yếu, dùng cách chunking như mã cũ của bạn.
    print("   Loading embeddings into RAM (fast method)...")
    df_emb = pl.read_parquet(FILE_EMBED).sort("new_id")

    # Kiểm tra cột vector tên gì
    col_name = "embedding" if "embedding" in df_emb.columns else "embed"

    # Convert sang Numpy Matrix
    matrix = np.stack(df_emb[col_name].to_numpy())

    # Save
    np.save(OUTPUT_DIR / "embeddings.npy", matrix)
    print(f"   ✅ Saved embeddings.npy {matrix.shape}")

    del df_emb, matrix
    gc.collect()

    # ---------------------------------------------------------
    # 2. ARTIST MAP (Sparse -> Dense Array)
    # ---------------------------------------------------------
    print("\n[2/3] 🎤 Processing Artist Map...")

    # Tạo mảng chứa toàn số 0 (Unknown)
    # Dùng int32 để tiết kiệm (nếu < 2 tỷ artist)
    artist_dense = np.zeros(N_ITEMS, dtype=np.int32)

    if FILE_ARTIST.exists():
        df_art = pl.read_parquet(FILE_ARTIST)
        # item_id chính là index trong mảng dense
        indices = df_art["item_id"].to_numpy()
        values = df_art["artist_id"].to_numpy()

        # Gán giá trị
        artist_dense[indices] = values
        count = len(indices)
    else:
        print("   ⚠️ No artist file found, array will be zeros.")
        count = 0

    np.save(OUTPUT_DIR / "artists.npy", artist_dense)
    print(f"   ✅ Saved artists.npy (Filled {count:,}/{N_ITEMS:,} items)")
    del artist_dense

    # ---------------------------------------------------------
    # 3. ALBUM MAP (Sparse -> Dense Array)
    # ---------------------------------------------------------
    print("\n[3/3] 💿 Processing Album Map...")

    album_dense = np.zeros(N_ITEMS, dtype=np.int32)

    if FILE_ALBUM.exists():
        df_alb = pl.read_parquet(FILE_ALBUM)
        indices = df_alb["item_id"].to_numpy()
        values = df_alb["album_id"].to_numpy()

        album_dense[indices] = values
        count = len(indices)
    else:
        print("   ⚠️ No album file found, array will be zeros.")
        count = 0

    np.save(OUTPUT_DIR / "albums.npy", album_dense)
    print(f"   ✅ Saved albums.npy (Filled {count:,}/{N_ITEMS:,} items)")
    del album_dense

    print("\n🎉 Static Data Build Complete!")

def build_interactions():
    """
    Xây dựng cấu trúc dữ liệu tương tác (Interactions) tối ưu cho Random Access.
    
    Architecture:
    - Sharding: Chia nhỏ dữ liệu thành nhiều partition dựa trên User ID hash để xử lý song song và tránh tràn RAM.
    - Sorting: Sắp xếp dữ liệu theo (User, Time) để đảm bảo tính tuần tự thời gian.
    - Flattening: Gộp tất cả partition thành 2 mảng phẳng khổng lồ (Items, Timestamps) lưu trên đĩa (Memory Mapped).
    - Indexing: Tạo mảng `offsets` để trỏ đến vị trí bắt đầu/kết thúc của từng User trong mảng phẳng.
    
    Kết quả: Truy xuất lịch sử của User bất kỳ chỉ tốn O(1) disk seek.
    """
    
    cfg = TrainingConfig()
    
    # CONFIGURATION
    INPUT_LISTENS = cfg.DATA_ROOT / "remapped_data/listens.parquet"
    OUTPUT_DIR = cfg.INTERACTIONS_DIR
    TEMP_DIR = cfg.DATA_ROOT / "temp_partitions"

    # CẤU HÌNH TÊN CỘT (Đã sửa theo log lỗi của bạn)
    COL_USER = "uid"        # <--- SỬA TỪ 'user_id' THÀNH 'uid'
    COL_ITEM = "item_id"
    COL_TIME = "timestamp"

    NUM_PARTITIONS = 50

    def check_schema():
        """Kiểm tra tên cột trước khi chạy để tránh lỗi giữa chừng"""
        print("🔍 Checking Schema...")
        try:
            schema = pl.scan_parquet(INPUT_LISTENS).limit(1).collect().columns
            print(f"   Detected Columns: {schema}")

            required = [COL_USER, COL_ITEM, COL_TIME]
            missing = [col for col in required if col not in schema]

            if missing:
                print(f"❌ ERROR: Missing columns in parquet file: {missing}")
                print(f"   Please update CONFIG variables in the script.")
                return False
            return True
        except Exception as e:
            print(f"❌ ERROR reading file: {e}")
            return False

    # 1. Check Schema trước
    if not check_schema():
        return

    # 2. Setup thư mục
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🚀 BUILDING INTERACTION ARRAYS (CSR Style)")
    print("="*60)

    # ---------------------------------------------------------
    # PHASE 1: SHARDING (Chia nhỏ file để sort)
    # ---------------------------------------------------------
    print("\n[1/3] 🔪 Partitioning data by User ID...")

    # Đếm tổng dòng
    total_rows = pl.scan_parquet(INPUT_LISTENS).select(pl.count()).collect().item()
    print(f"   Total Interactions: {total_rows:,}")

    # Loop qua các partition
    for i in tqdm(range(NUM_PARTITIONS), desc="Partitioning"):
        # Filter & Select
        lf = pl.scan_parquet(INPUT_LISTENS).filter(
            (pl.col(COL_USER).hash(seed=42) % NUM_PARTITIONS) == i
        )

        # Select cột cần thiết và cast kiểu
        df_part = lf.select([
            pl.col(COL_USER).cast(pl.UInt32),
            pl.col(COL_ITEM).cast(pl.UInt32),
            pl.col(COL_TIME).cast(pl.UInt32),
        ]).collect()

        if df_part.height > 0:
            # Sort ngay tại đây (Quan trọng: Sort theo UID trước, rồi đến Time)
            df_part = df_part.sort([COL_USER, COL_TIME])
            df_part.write_parquet(TEMP_DIR / f"part_{i:03d}.parquet")

        del df_part
        gc.collect()

    # ---------------------------------------------------------
    # PHASE 2: ALLOCATE MEMMAP (Tạo file rỗng trên đĩa)
    # ---------------------------------------------------------
    print("\n[2/3] 💾 Allocating memory-mapped files...")

    # Pre-allocate file kích thước lớn trên ổ cứng
    mmap_items = np.memmap(OUTPUT_DIR / "flat_item_ids.npy", dtype='uint32', mode='w+', shape=(total_rows,))
    mmap_times = np.memmap(OUTPUT_DIR / "flat_timestamps.npy", dtype='uint32', mode='w+', shape=(total_rows,))

    # List tạm chứa độ dài lịch sử của từng user
    user_lengths = []

    # ---------------------------------------------------------
    # PHASE 3: MERGE & FLATTEN
    # ---------------------------------------------------------
    print("\n[3/3] 🚜 Merging partitions into flat arrays...")

    current_offset = 0
    partition_files = sorted(TEMP_DIR.glob("*.parquet"))

    for p_file in tqdm(partition_files, desc="Merging"):
        df = pl.read_parquet(p_file)

        # Lấy mảng numpy ra (Siêu nhanh)
        arr_users = df[COL_USER].to_numpy()
        arr_items = df[COL_ITEM].to_numpy()
        arr_times = df[COL_TIME].to_numpy()

        # 1. Copy Data vào Mmap
        n_rows = len(df)
        mmap_items[current_offset : current_offset + n_rows] = arr_items
        mmap_times[current_offset : current_offset + n_rows] = arr_times
        current_offset += n_rows

        # 2. Tính User Group Lengths
        # Vì data đã sort theo user, ta dùng np.unique để đếm số dòng của mỗi user
        # return_counts trả về số interaction của từng user
        # Lưu ý: Vì ta chia partition theo hash user, nên 1 user CHẮC CHẮN chỉ nằm trọn vẹn trong 1 partition
        _, counts = np.unique(arr_users, return_counts=True)

        # counts chính là length history của từng user trong partition này
        user_lengths.extend(counts)

        del df, arr_users, arr_items, arr_times
        gc.collect()

    # Flush data xuống đĩa (Save)
    mmap_items.flush()
    mmap_times.flush()

    # Tính User Offsets
    print("   Calculating User Offsets...")
    user_lengths = np.array(user_lengths, dtype=np.uint32)

    # Offsets là mảng tích lũy: [0, len_u1, len_u1+len_u2, ...]
    offsets = np.zeros(len(user_lengths) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(user_lengths)

    np.save(OUTPUT_DIR / "user_offsets.npy", offsets)

    print(f"✅ DONE!")
    print(f"   Total Users : {len(user_lengths):,}")
    print(f"   Output Files: {OUTPUT_DIR}")

    # Xóa temp để giải phóng ổ cứng
    shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    process_data()
    build_static_data()
    build_interactions()
