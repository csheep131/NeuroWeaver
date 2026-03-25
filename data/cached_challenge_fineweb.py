"""Cached FineWeb Dataset Loader for Parameter Golf Challenge.

This script downloads and caches the FineWeb dataset with a 1024-token vocabulary
for the Parameter Golf Challenge. The dataset is pre-tokenized and stored in
binary shards for efficient loading during training.

Challenge Requirements:
- FineWeb validation set (first 50k documents)
- 1024-token vocabulary (SentencePiece BPE)
- Pre-tokenized binary shards for fast loading

Usage:
    # Download full validation set + 80 training shards (8B tokens)
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
    
    # Download smaller subset for testing
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
    
    # Download validation set only
    python data/cached_challenge_fineweb.py --variant sp1024 --val-only

Environment Variables:
    HF_TOKEN: HuggingFace token (optional, for faster downloads)
    DATA_CACHE: Custom cache directory (default: ./data/datasets/)
    TOKENIZER_CACHE: Custom tokenizer directory (default: ./data/tokenizers/)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install datasets tqdm sentencepiece")
    sys.exit(1)

try:
    import sentencepiece as spm
except ImportError:
    print("Warning: sentencepiece not installed. Tokenizer training will fail.")
    spm = None


# ============================================================================
# Configuration
# ============================================================================

DEFAULTS = {
    "fineweb_url": "HuggingFaceFW/fineweb",
    "fineweb_sample": "CC-MAIN-2024-10",  # Specific sample for reproducibility
    "val_size": 50_000,  # 50k documents for validation
    "tokens_per_document": 2048,  # Average tokens per document
    "bytes_per_token": 2,  # uint16 storage
}


def get_cache_dirs() -> dict:
    """Get cache directories."""
    data_cache = os.getenv("DATA_CACHE", "./data/datasets/")
    tokenizer_cache = os.getenv("TOKENIZER_CACHE", "./data/tokenizers/")
    
    return {
        "data": Path(data_cache),
        "tokenizer": Path(tokenizer_cache),
    }


# ============================================================================
# Tokenizer
# ============================================================================

class SentencePieceTokenizer:
    """SentencePiece tokenizer for FineWeb."""
    
    def __init__(self, vocab_size: int = 1024, model_prefix: str = "fineweb"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp = None
    
    def train(self, text_iterator, output_dir: Path):
        """Train SentencePiece tokenizer from text iterator."""
        if spm is None:
            raise ImportError("sentencepiece is required for tokenizer training")
        
        # Create temporary file for training data
        train_file = output_dir / "train_text.txt"
        train_file.parent.mkdir(parents=True, exist_ok=True)
        
        print("Collecting training data for tokenizer...")
        with open(train_file, "w", encoding="utf-8") as f:
            count = 0
            for text in tqdm(text_iterator, desc="Writing training data"):
                # SentencePiece expects one document per line
                f.write(text.replace("\n", " ") + "\n")
                count += 1
                if count >= 100_000:  # Use 100k documents for training
                    break
        
        print(f"Training tokenizer on {count} documents...")
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=str(train_file),
            model_prefix=str(output_dir / self.model_prefix),
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            minloglevel=0,
        )
        
        # Cleanup temporary file
        train_file.unlink()
        
        print(f"Tokenizer trained and saved to {output_dir}")
        
        return output_dir / f"{self.model_prefix}.model"
    
    def load(self, model_path: Path):
        """Load trained tokenizer."""
        if spm is None:
            raise ImportError("sentencepiece is required")
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))
        print(f"Tokenizer loaded: vocab_size={self.sp.GetPieceSize()}")
        
        return self.sp
    
    def encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.EncodeAsIds(text)
    
    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.DecodeIds(tokens)


# ============================================================================
# Dataset Download & Preprocessing
# ============================================================================

def download_fineweb(split: str = "train", shards: int = 80) -> list:
    """Download FineWeb dataset.
    
    Args:
        split: 'train' or 'validation'
        shards: Number of training shards to download (for train split)
    
    Returns:
        List of documents
    """
    print(f"Downloading FineWeb {split} split...")
    
    # Load dataset from HuggingFace
    if split == "validation":
        # Validation: fixed 50k documents
        dataset = load_dataset(
            DEFAULTS["fineweb_url"],
            name=DEFAULTS["fineweb_sample"],
            split="validation",
            trust_remote_code=True,
        )
        
        # Take first 50k documents
        dataset = dataset.select(range(min(DEFAULTS["val_size"], len(dataset))))
        
    else:
        # Training: download specified number of shards
        # Each shard is approximately 100M tokens
        dataset = load_dataset(
            DEFAULTS["fineweb_url"],
            name=DEFAULTS["fineweb_sample"],
            split=f"train[:{shards}*100M]",
            trust_remote_code=True,
        )
    
    print(f"Loaded {len(dataset)} documents")
    
    return dataset


def tokenize_and_save(dataset, tokenizer, output_path: Path, shard_size: int = 100_000_000):
    """Tokenize dataset and save to binary shards.
    
    Args:
        dataset: FineWeb dataset
        tokenizer: SentencePiece tokenizer
        output_path: Output directory for shards
        shard_size: Tokens per shard
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_tokens = []
    shard_idx = 0
    
    print(f"Tokenizing dataset (shard_size={shard_size:,} tokens)...")
    
    for doc in tqdm(dataset, desc="Tokenizing"):
        text = doc.get("text", "")
        if not text:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        
        # Save shard when full
        while len(all_tokens) >= shard_size:
            shard_tokens = all_tokens[:shard_size]
            all_tokens = all_tokens[shard_size:]
            
            # Save to binary file
            shard_path = output_path / f"shard_{shard_idx:05d}.bin"
            arr = np.array(shard_tokens, dtype=np.uint16)
            arr.tofile(shard_path)
            
            shard_idx += 1
    
    # Save remaining tokens
    if all_tokens:
        shard_path = output_path / f"shard_{shard_idx:05d}.bin"
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(shard_path)
        shard_idx += 1
    
    print(f"Saved {shard_idx} shards to {output_path}")
    
    return shard_idx


# ============================================================================
# Main Functions
# ============================================================================

def prepare_tokenizer(variant: str = "sp1024") -> Path:
    """Prepare tokenizer for challenge.
    
    Args:
        variant: Tokenizer variant (sp1024 = SentencePiece 1024 vocab)
    
    Returns:
        Path to tokenizer model
    """
    dirs = get_cache_dirs()
    
    # Parse variant
    if variant.startswith("sp"):
        vocab_size = int(variant[2:])
    else:
        vocab_size = 1024
    
    model_prefix = f"fineweb_{vocab_size}_bpe"
    model_path = dirs["tokenizer"] / f"{model_prefix}.model"
    
    # Check if already exists
    if model_path.exists():
        print(f"Tokenizer already exists: {model_path}")
        return model_path
    
    # Train new tokenizer
    print(f"Training new tokenizer (vocab_size={vocab_size})...")
    
    # Download sample data for training
    sample_dataset = download_fineweb(split="train", shards=1)
    
    tokenizer = SentencePieceTokenizer(vocab_size=vocab_size, model_prefix=model_prefix)
    tokenizer.train(
        text_iterator=(doc["text"] for doc in sample_dataset),
        output_dir=dirs["tokenizer"],
    )
    
    return model_path


def prepare_dataset(
    variant: str = "sp1024",
    train_shards: int = 80,
    val_only: bool = False,
) -> dict:
    """Prepare FineWeb dataset for challenge.
    
    Args:
        variant: Tokenizer variant
        train_shards: Number of training shards
        val_only: Download validation set only
    
    Returns:
        Dictionary with dataset paths
    """
    dirs = get_cache_dirs()
    
    # Prepare tokenizer
    tokenizer_path = prepare_tokenizer(variant)
    
    # Load tokenizer
    tokenizer = SentencePieceTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Parse variant for output directory
    if variant.startswith("sp"):
        vocab_size = int(variant[2:])
    else:
        vocab_size = 1024
    
    dataset_name = f"fineweb{vocab_size // 1024}B_{variant}"
    dataset_dir = dirs["data"] / dataset_name
    
    # Download and tokenize validation set
    val_dir = dataset_dir / "val"
    if not val_dir.exists():
        print("Downloading validation set...")
        val_dataset = download_fineweb(split="validation")
        tokenize_and_save(val_dataset, tokenizer, val_dir, shard_size=10_000_000)
    else:
        print(f"Validation set already exists: {val_dir}")
    
    # Download and tokenize training set
    if not val_only:
        train_dir = dataset_dir / "train"
        if not train_dir.exists():
            print(f"Downloading training set ({train_shards} shards)...")
            train_dataset = download_fineweb(split="train", shards=train_shards)
            tokenize_and_save(train_dataset, tokenizer, train_dir, shard_size=100_000_000)
        else:
            print(f"Training set already exists: {train_dir}")
    
    return {
        "tokenizer_path": str(tokenizer_path),
        "train_path": str(train_dir) if not val_only else None,
        "val_path": str(val_dir),
        "dataset_dir": str(dataset_dir),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and cache FineWeb dataset for Parameter Golf Challenge"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="sp1024",
        help="Tokenizer variant (e.g., sp1024 for SentencePiece 1024 vocab)",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=80,
        help="Number of training shards to download (default: 80 = 8B tokens)",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Download validation set only",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing cache before downloading",
    )
    
    args = parser.parse_args()
    
    # Clean cache if requested
    if args.clean:
        dirs = get_cache_dirs()
        for d in [dirs["data"], dirs["tokenizer"]]:
            if d.exists():
                print(f"Cleaning {d}...")
                shutil.rmtree(d)
    
    # Prepare dataset
    import numpy as np  # Import here to avoid error if not installed yet
    
    result = prepare_dataset(
        variant=args.variant,
        train_shards=args.train_shards,
        val_only=args.val_only,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete")
    print("=" * 60)
    print(f"Tokenizer: {result['tokenizer_path']}")
    print(f"Validation: {result['val_path']}")
    if result['train_path']:
        print(f"Training: {result['train_path']}")
    print(f"Dataset directory: {result['dataset_dir']}")
    print("=" * 60)
    
    # Usage instructions
    print("\nUsage in train_gpt.py:")
    print(f"  DATA_PATH={result['train_path'] or result['val_path']} \\")
    print(f"  TOKENIZER_PATH={result['tokenizer_path']} \\")
    print(f"  VOCAB_SIZE=1024 \\")
    print(f"  python train_gpt.py")


if __name__ == "__main__":
    main()
