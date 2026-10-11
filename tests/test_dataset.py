"""
Unit test label mapping dan dataset structure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess_dataset import build_label_map, discover_videos


# ---------------------------------------------------------------------------
# Test: build_label_map
# ---------------------------------------------------------------------------

class TestBuildLabelMap:
    def test_basic_mapping(self):
        labels = ["kamu_makan", "terima_kasih", "menelepon"]
        lbl2idx, idx2lbl = build_label_map(labels)
        assert set(lbl2idx.keys()) == {"kamu_makan", "terima_kasih", "menelepon"}
        assert set(lbl2idx.values()) == {0, 1, 2}

    def test_sorted_order(self):
        """Mapping harus deterministic (sorted)."""
        labels = ["c", "a", "b"]
        lbl2idx, _ = build_label_map(labels)
        assert lbl2idx["a"] == 0
        assert lbl2idx["b"] == 1
        assert lbl2idx["c"] == 2

    def test_duplicates_deduplicated(self):
        labels = ["kamu_makan", "kamu_makan", "menelepon"]
        lbl2idx, _ = build_label_map(labels)
        assert len(lbl2idx) == 2

    def test_inverse_mapping_consistent(self):
        labels = ["kamu_makan", "terima_kasih"]
        lbl2idx, idx2lbl = build_label_map(labels)
        for lbl, idx in lbl2idx.items():
            assert idx2lbl[idx] == lbl

    def test_single_class(self):
        labels = ["satu_kelas"]
        lbl2idx, idx2lbl = build_label_map(labels)
        assert lbl2idx["satu_kelas"] == 0
        assert idx2lbl[0] == "satu_kelas"

    def test_consistent_across_calls(self):
        """Mapping harus sama jika input sama."""
        labels = ["b", "a", "c"]
        m1, _ = build_label_map(labels)
        m2, _ = build_label_map(labels)
        assert m1 == m2

    def test_index_range_zero_based(self):
        labels = ["x", "y", "z", "w"]
        lbl2idx, _ = build_label_map(labels)
        assert min(lbl2idx.values()) == 0
        assert max(lbl2idx.values()) == 3


# ---------------------------------------------------------------------------
# Test: discover_videos
# ---------------------------------------------------------------------------

class TestDiscoverVideos:
    def _make_fake_dataset(self, tmp_path: Path, structure: dict) -> Path:
        """Buat struktur folder palsu untuk test discovery."""
        root = tmp_path / "raw_videos"
        for class_name, files in structure.items():
            cls_dir = root / class_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for fname in files:
                (cls_dir / fname).write_bytes(b"fake")
        return root

    def test_discovers_all_classes(self, tmp_path):
        structure = {
            "kamu_makan": ["s001.mp4", "s002.mp4"],
            "menelepon": ["s001.mp4"],
            "terima_kasih": ["s001.mp4", "s002.mp4", "s003.mp4"],
        }
        root = self._make_fake_dataset(tmp_path, structure)
        samples = discover_videos(root)
        labels = {lbl for _, lbl in samples}
        assert labels == {"kamu_makan", "menelepon", "terima_kasih"}

    def test_total_sample_count(self, tmp_path):
        structure = {
            "kamu_makan": ["s001.mp4", "s002.mp4"],
            "menelepon": ["s001.mp4"],
        }
        root = self._make_fake_dataset(tmp_path, structure)
        samples = discover_videos(root)
        assert len(samples) == 3

    def test_ignores_non_video_files(self, tmp_path):
        structure = {
            "kamu_makan": ["s001.mp4", "notes.txt", "image.png"],
        }
        root = self._make_fake_dataset(tmp_path, structure)
        samples = discover_videos(root)
        assert len(samples) == 1  # hanya mp4

    def test_supported_extensions(self, tmp_path):
        structure = {
            "kelas_a": ["vid.mp4", "vid.avi", "vid.mov", "vid.mkv"],
        }
        root = self._make_fake_dataset(tmp_path, structure)
        samples = discover_videos(root)
        assert len(samples) == 4

    def test_empty_class_warns(self, tmp_path, capsys):
        structure = {
            "dengan_video": ["s001.mp4"],
            "tanpa_video": [],  # folder kosong
        }
        root = self._make_fake_dataset(tmp_path, structure)
        samples = discover_videos(root)
        # Kelas tanpa video tidak menghasilkan sampel
        assert all(lbl == "dengan_video" for _, lbl in samples)

    def test_no_classes_raises(self, tmp_path):
        empty_root = tmp_path / "empty_raw"
        empty_root.mkdir()
        with pytest.raises(FileNotFoundError, match="Tidak ada subfolder kelas"):
            discover_videos(empty_root)

    def test_label_from_folder_name(self, tmp_path):
        structure = {"terima_kasih": ["s001.mp4"]}
        root = self._make_fake_dataset(tmp_path, structure)
        samples = discover_videos(root)
        assert samples[0][1] == "terima_kasih"
