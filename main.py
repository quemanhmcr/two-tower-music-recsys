import torch
import gc
from torch.utils.data import DataLoader

from config import TrainingConfig
from utils.common import get_vocab_sizes_from_npy
from utils.scheduling import get_cosine_schedule_with_warmup
from data.dataset import RichFeatureDataset
from model.two_tower import TwoTowerModel
from training.trainer import TwoTowerTrainer
from inference.index import generate_and_index
from inference.retrieve import test_retrieval

def create_model(cfg: TrainingConfig):
    """Create Two-Tower model using the config."""
    print("\nCreating model...")

    num_artists, num_albums, num_items = get_vocab_sizes_from_npy(
        cfg.ARTIST_MAP_FILE,
        cfg.ALBUM_MAP_FILE,
        cfg.EMBEDDINGS_FILE
    )

    user_tower_config = {
        'embed_dim': cfg.EMBED_DIM, 'hidden_dims': cfg.HIDDEN_DIMS_USER, 'dropout': 0.2,
        'num_eng_buckets': 5, 'num_time_slots': 4, 'max_seq_len': cfg.MAX_SEQ_LEN,
        'use_layer_norm': True, 'activation': 'relu', 'num_items': num_items
    }
    item_tower_config = {
        'embed_dim': cfg.EMBED_DIM, 'hidden_dims': cfg.HIDDEN_DIMS_ITEM, 'dropout': 0.2,
        'num_artists': num_artists, 'num_albums': num_albums, "num_items": num_items,
        'artist_emb_dim': 32, 'album_emb_dim': 32,
        'use_layer_norm': True, 'activation': 'relu'
    }

    model = TwoTowerModel(
        user_tower_config=user_tower_config,
        item_tower_config=item_tower_config,
        temperature=cfg.TEMPERRATURE,
        use_learnable_temp=cfg.USE_LEARNABLE,
    )
    return model

def main():
    """
    Main Entry Point của Training Pipeline.
    
    Orchestration Flow:
    1. Config Init: Load cấu hình toàn cục.
    2. Data Loading: Khởi tạo Dataset & DataLoader (Multi-worker).
    3. Model Init: Xây dựng kiến trúc Two-Tower dựa trên Vocab Size thực tế.
    4. Training Loop: Chạy vòng lặp huấn luyện với Trainer.
    5. Inference & Indexing: Sinh vector cho toàn bộ Item và tạo FAISS Index.
    6. Evaluation: Kiểm tra thử khả năng truy xuất của model.
    """
    # 1. Init Config
    config = TrainingConfig()

    # 2. Create Datasets
    print("Loading datasets...")
    train_ds = RichFeatureDataset(config, mode="train")
    val_ds = RichFeatureDataset(config, mode="val")

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None,
        persistent_workers=(config.NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # 3. CRITICAL RAM HACK:
    print(f"🧹 Cleaning up RAM before model creation...")
    del train_ds, val_ds
    gc.collect()
    print("   Dataset variables deleted from global scope.")

    # 4. Create Model
    model = create_model(config)

    # 5. Setup Optimizer with Weight Decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=config.LR,
        betas=(0.9, 0.98), eps=1e-6
    )

    # 6. Setup Scheduler
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = config.NUM_EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(config.WARMUP_RATIO * max_train_steps)
    print(f"Total steps: {max_train_steps}, Warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )

    # 7. Initialize Trainer
    trainer = TwoTowerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_amp=True,
        log_interval=500,
        accumulation_steps=config.ACCMULATION_STEPS,
        val_interval=config.VAL_INTERVAL,
        val_batches_limit=config.VAL_BATCHES_LIMIT,
    )

    # 8. Start Training
    print("\n🔥 STARTING TRAINING 🔥")
    trainer.fit(num_epochs=config.NUM_EPOCHS)

    # 9. Inference & Indexing
    try:
        faiss_index, vectors = generate_and_index(trainer, config)

        print("\n🔎 SANITY CHECK:")
        query = vectors[0:1]
        D, I = faiss_index.search(query, k=3)
        print(f"   Query Item Index: 0 -> Top 1 Result: {I[0][0]} (Should be 0)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # 10. Test Retrieval
    test_retrieval(trainer, config)

if __name__ == "__main__":
    main()
