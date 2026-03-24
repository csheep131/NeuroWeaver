"""Daten-Loader für Text-Training.

Dieses Modul lädt und tokenisiert Textdaten für das Training
der Ablation Machine Modelle.

Features:
- Lädt Textdateien (UTF-8)
- Tokenisierung (Byte, Bigram, Trigram)
- Batching mit konfigurierbarer Sequenzlänge
- Shuffle für Training
- Streaming für große Datensätze
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Protocol

import numpy as np


class TokenizerProtocol(Protocol):
    """Protocol für Tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens to text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


@dataclass
class TextDataConfig:
    """Konfiguration für Daten-Loader."""

    # Daten-Pfade
    train_data_path: str = ""
    eval_data_path: str = ""

    # Sequenz-Parameter
    seq_len: int = 256
    batch_size: int = 32

    # Tokenizer
    tokenizer_type: str = "byte"
    tokenizer_vocab_size: int = 256

    # Training
    shuffle: bool = True
    seed: int = 42

    # Streaming
    streaming: bool = False
    buffer_size: int = 1000

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextDataConfig":
        """Create from dictionary."""
        return cls(
            train_data_path=d.get("train_data_path", ""),
            eval_data_path=d.get("eval_data_path", ""),
            seq_len=d.get("seq_len", 256),
            batch_size=d.get("batch_size", 32),
            tokenizer_type=d.get("tokenizer_type", "byte"),
            tokenizer_vocab_size=d.get("tokenizer_vocab_size", 256),
            shuffle=d.get("shuffle", True),
            seed=d.get("seed", 42),
            streaming=d.get("streaming", False),
            buffer_size=d.get("buffer_size", 1000),
        )


@dataclass
class Batch:
    """Ein Batch von Trainingsdaten."""

    # Token-IDs: (batch_size, seq_len)
    tokens: np.ndarray

    # Target-IDs: (batch_size, seq_len) - um 1 verschoben
    targets: np.ndarray

    # Attention-Mask: (batch_size, seq_len) - 1 für echte Tokens, 0 für Padding
    mask: np.ndarray

    # Anzahl Bytes im Originaltext
    num_bytes: int = 0

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.tokens.shape[0]

    @property
    def seq_len(self) -> int:
        """Get sequence length."""
        return self.tokens.shape[1]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens": self.tokens,
            "targets": self.targets,
            "mask": self.mask,
            "num_bytes": self.num_bytes,
        }


class TextDataLoader:
    """Daten-Loader für Text-Training.

    Lädt Textdateien, tokenisiert und erstellt Batches.
    """

    def __init__(
        self,
        config: TextDataConfig,
        tokenizer: TokenizerProtocol | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(config.seed)

        # Daten-Cache (für nicht-streaming Modus)
        self._tokens_cache: np.ndarray | None = None

    def load_data(self, data_path: str) -> np.ndarray:
        """Lade und tokenisiere Daten.

        Args:
            data_path: Pfad zu Textdatei oder Verzeichnis

        Returns:
            Token-Array (flattened)
        """
        path = Path(data_path)

        if not path.exists():
            # Generate synthetic data if file doesn't exist
            return self._generate_synthetic_data()

        if path.is_file():
            return self._load_file(path)
        elif path.is_dir():
            return self._load_directory(path)
        else:
            raise ValueError(f"Invalid data path: {data_path}")

    def _load_file(self, path: Path) -> np.ndarray:
        """Lade einzelne Textdatei."""
        all_tokens = []

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenisieren
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
        else:
            # Byte-level Fallback
            tokens = [b for b in text.encode("utf-8")]

        all_tokens.extend(tokens)

        return np.array(all_tokens, dtype=np.int64)

    def _load_directory(self, path: Path) -> np.ndarray:
        """Lade alle Textdateien in Verzeichnis."""
        all_tokens = []

        # Finde alle .txt Dateien
        txt_files = list(path.glob("**/*.txt"))

        if not txt_files:
            return self._generate_synthetic_data()

        for txt_file in txt_files:
            tokens = self._load_file(txt_file)
            all_tokens.extend(tokens)

        return np.array(all_tokens, dtype=np.int64)

    def _generate_synthetic_data(self) -> np.ndarray:
        """Generiere synthetische Trainingsdaten.

        Wird verwendet wenn keine echten Daten vorhanden sind.
        Generiert zufälligen Text mit natürlicher Token-Verteilung.
        """
        # Generiere ~1MB an zufälligem Text
        num_chars = 1_000_000

        # Verwende häufige ASCII-Zeichen für natürlichere Verteilung
        chars = list(range(32, 127))  # Druckbare ASCII-Zeichen
        weights = np.random.dirichlet(np.ones(len(chars)))  # Natürliche Verteilung

        tokens = self.rng.choice(chars, size=num_chars, p=weights)
        return tokens.astype(np.int64)

    def create_batches(
        self,
        tokens: np.ndarray,
        shuffle: bool | None = None,
    ) -> Iterator[Batch]:
        """Erstelle Batches aus Tokens.

        Args:
            tokens: Flattened Token-Array
            shuffle: Ob shuffeln (default: config.shuffle)

        Yields:
            Batch-Objekte
        """
        if shuffle is None:
            shuffle = self.config.shuffle

        seq_len = self.config.seq_len
        batch_size = self.config.batch_size

        # Trimme auf vielfache von seq_len
        num_tokens = len(tokens)
        num_complete_seqs = num_tokens // seq_len
        trimmed_tokens = tokens[: num_complete_seqs * seq_len]

        # Reshape zu Sequenzen
        sequences = trimmed_tokens.reshape(-1, seq_len)

        # Shuffle
        if shuffle:
            indices = self.rng.permutation(len(sequences))
            sequences = sequences[indices]

        # Erstelle Batches
        num_batches = len(sequences) // batch_size

        for i in range(num_batches):
            batch_seqs = sequences[i * batch_size : (i + 1) * batch_size]

            # Tokens und Targets (um 1 verschoben)
            batch_tokens = batch_seqs[:, :-1]
            batch_targets = batch_seqs[:, 1:]

            # Padding-Mask (alle 1 da wir getrimmt haben)
            mask = np.ones_like(batch_tokens, dtype=np.float32)

            # Anzahl Bytes
            num_bytes = batch_tokens.size

            yield Batch(
                tokens=batch_tokens,
                targets=batch_targets,
                mask=mask,
                num_bytes=num_bytes,
            )

    def __iter__(self) -> Iterator[Batch]:
        """Iteriere über Trainingsdaten."""
        if self.config.streaming:
            return self._stream_batches()
        else:
            return self._load_all_batches()

    def _load_all_batches(self) -> Iterator[Batch]:
        """Lade alle Daten in den Speicher und erstelle Batches."""
        # Cache laden oder erstellen
        if self._tokens_cache is None:
            data_path = self.config.train_data_path
            if data_path:
                self._tokens_cache = self.load_data(data_path)
            else:
                self._tokens_cache = self._generate_synthetic_data()

        return self.create_batches(self._tokens_cache)

    def _stream_batches(self) -> Iterator[Batch]:
        """Stream Batches aus Datei (für große Datensätze)."""
        data_path = self.config.train_data_path

        if not data_path or not Path(data_path).exists():
            # Fallback zu synthetic data
            tokens = self._generate_synthetic_data()
            yield from self.create_batches(tokens)
            return

        # Lese Datei in Chunks
        buffer = []
        buffer_size = self.config.buffer_size * self.config.seq_len

        path = Path(data_path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                while True:
                    text = f.read(buffer_size)
                    if not text:
                        break

                    # Tokenisieren
                    if self.tokenizer:
                        chunk_tokens = self.tokenizer.encode(text)
                    else:
                        chunk_tokens = [b for b in text.encode("utf-8")]

                    buffer.extend(chunk_tokens)

                    # Erstelle Batches wenn Buffer voll
                    while len(buffer) >= self.config.seq_len * self.config.batch_size:
                        batch_tokens = buffer[: self.config.seq_len * self.config.batch_size]
                        buffer = buffer[self.config.seq_len * self.config.batch_size :]

                        tokens_array = np.array(batch_tokens, dtype=np.int64)
                        yield from self.create_batches(tokens_array, shuffle=False)

    def get_eval_loader(
        self,
        eval_size: int | None = None,
    ) -> Iterator[Batch]:
        """Erstelle Eval-Loader.

        Args:
            eval_size: Anzahl Tokens für Evaluation (default: config.seq_len * 100)

        Yields:
            Batch-Objekte für Evaluation
        """
        if eval_size is None:
            eval_size = self.config.seq_len * 100

        # Lade Eval-Daten oder verwende Trainingsdaten
        eval_path = self.config.eval_data_path
        if eval_path and Path(eval_path).exists():
            tokens = self.load_data(eval_path)
        else:
            # Verwende Teil der Trainingsdaten
            if self._tokens_cache is None:
                self._tokens_cache = self.load_data(self.config.train_data_path)
            tokens = self._tokens_cache[:eval_size]

        # Nicht shuffeln für Evaluation
        yield from self.create_batches(tokens, shuffle=False)


def create_dataloader(
    config: dict[str, Any] | TextDataConfig,
    tokenizer: TokenizerProtocol | None = None,
) -> TextDataLoader:
    """Erstelle Daten-Loader aus Konfiguration.

    Args:
        config: Konfiguration oder Dictionary
        tokenizer: Optionaler Tokenizer

    Returns:
        TextDataLoader Instanz
    """
    if isinstance(config, dict):
        config = TextDataConfig.from_dict(config)

    return TextDataLoader(config, tokenizer)


def create_tokenizer(
    tokenizer_type: str = "byte",
    vocab_size: int = 256,
    byte_fallback: bool = True,
) -> Any:
    """Erstelle Tokenizer für Daten-Loader.

    Args:
        tokenizer_type: "byte", "bigram_hash", "trigram_hash"
        vocab_size: Vokabulargröße
        byte_fallback: Byte-Fallback aktivieren

    Returns:
        Tokenizer-Instanz
    """
    # Importiere Tokenizer aus tokenizers Modul
    try:
        from tokenizers.tokenizers import (
            ByteTokenizer,
            BigramHashTokenizer,
            TrigramHashTokenizer,
        )

        if tokenizer_type == "byte":
            return ByteTokenizer(vocab_size)
        elif tokenizer_type == "bigram_hash":
            return BigramHashTokenizer(vocab_size, byte_fallback)
        elif tokenizer_type == "trigram_hash":
            return TrigramHashTokenizer(vocab_size, byte_fallback)
        else:
            return ByteTokenizer(vocab_size)

    except ImportError:
        # Fallback: Einfacher Byte-Tokenizer
        class SimpleByteTokenizer:
            def __init__(self, vocab_size: int = 256):
                self.vocab_size = min(vocab_size, 256)

            def encode(self, text: str) -> list[int]:
                return [b for b in text.encode("utf-8")]

            def decode(self, tokens: list[int]) -> str:
                try:
                    return bytes(tokens).decode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    return "?"

        return SimpleByteTokenizer(vocab_size)
