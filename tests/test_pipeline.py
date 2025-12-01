import unittest
import torch
import numpy as np
import shutil
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import TrainingConfig
from model.two_tower import TwoTowerModel
from training.trainer import TwoTowerTrainer
from data.dataset import RichFeatureDataset

class TestPipeline(unittest.TestCase):
    def setUp(self):
        """Setup dummy data for testing"""
        self.test_dir = Path("test_data_temp")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()

        # 1. Create Dummy Config
        self.cfg = TrainingConfig()
        self.cfg.DATA_ROOT = self.test_dir
        self.cfg.STATIC_DIR = self.test_dir / "static"
        self.cfg.INTERACTIONS_DIR = self.test_dir / "interactions"
        self.cfg.CHECKPOINT_DIR = str(self.test_dir / "checkpoints")
        
        self.cfg.STATIC_DIR.mkdir()
        self.cfg.INTERACTIONS_DIR.mkdir()
        Path(self.cfg.CHECKPOINT_DIR).mkdir()

        # 2. Create Dummy Static Data
        self.num_items = 100
        self.embed_dim = 32 # Small dim for test
        self.cfg.EMBED_DIM = self.embed_dim
        
        # Embeddings
        embeddings = np.random.randn(self.num_items, self.embed_dim).astype(np.float32)
        np.save(self.cfg.EMBEDDINGS_FILE, embeddings)
        
        # Artists/Albums
        artists = np.random.randint(0, 10, size=self.num_items).astype(np.int32)
        albums = np.random.randint(0, 10, size=self.num_items).astype(np.int32)
        np.save(self.cfg.ARTIST_MAP_FILE, artists)
        np.save(self.cfg.ALBUM_MAP_FILE, albums)

        # 3. Create Dummy Interactions
        num_interactions = 1000
        flat_items = np.random.randint(1, self.num_items, size=num_interactions).astype(np.uint32)
        flat_ts = np.arange(num_interactions).astype(np.uint32)
        
        # Save as mmap-able files
        fp_items = np.memmap(self.cfg.INTERACTIONS_DIR / "flat_item_ids.npy", dtype='uint32', mode='w+', shape=flat_items.shape)
        fp_items[:] = flat_items[:]
        fp_items.flush()
        
        fp_ts = np.memmap(self.cfg.INTERACTIONS_DIR / "flat_timestamps.npy", dtype='uint32', mode='w+', shape=flat_ts.shape)
        fp_ts[:] = flat_ts[:]
        fp_ts.flush()

        # Offsets (10 users, 100 items each)
        offsets = np.arange(0, num_interactions + 1, 100).astype(np.int64)
        np.save(self.cfg.INTERACTIONS_DIR / "user_offsets.npy", offsets)

    def tearDown(self):
        """Cleanup"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_training_loop(self):
        """Run 1 epoch of training with synthetic data"""
        print("\nTesting Training Loop...")
        
        # Init Dataset
        ds = RichFeatureDataset(self.cfg, mode="train")
        loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)

        # Init Model
        user_conf = {
            'embed_dim': self.embed_dim, 'hidden_dims': [32], 'num_items': self.num_items,
            'input_audio_dim': 128 # Default in code
        }
        item_conf = {
            'embed_dim': self.embed_dim, 'hidden_dims': [32], 'num_items': self.num_items,
            'num_artists': 10, 'num_albums': 10, 'input_audio_dim': 128
        }
        
        model = TwoTowerModel(user_conf, item_conf)
        
        # Init Trainer
        optimizer = torch.optim.Adam(model.parameters())
        trainer = TwoTowerTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader, # Reuse for test
            optimizer=optimizer,
            scheduler=None,
            device='cpu', # Force CPU for CI
            use_amp=False, # No AMP on CPU usually
            log_interval=1,
            val_interval=5,
            val_batches_limit=1
        )

        # Run 1 Epoch
        loss = trainer.train_epoch(epoch=1)
        self.assertIsInstance(loss, float)
        print(f"Test Loss: {loss}")

if __name__ == '__main__':
    unittest.main()
