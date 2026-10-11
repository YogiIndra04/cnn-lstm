"""Reset generated artifacts so the experiment can start from a clean state."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset generated data/model/report artifacts.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Hapus juga data/raw_videos (video mentah). Tanpa flag ini, video mentah tetap aman.",
    )
    parser.add_argument("--yes", action="store_true", help="Lewati konfirmasi interaktif.")
    return parser.parse_args()


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    targets = [
        Path("data/processed"),
        Path("data/splits"),
        Path("models"),
        Path("reports"),
    ]
    if args.full:
        targets.append(Path("data/raw_videos"))

    print("Folder yang akan dibersihkan:")
    for t in targets:
        print(f"- {t}")

    if not args.yes:
        ans = input("Lanjutkan reset? ketik 'YA' untuk konfirmasi: ").strip()
        if ans != "YA":
            print("Reset dibatalkan.")
            return

    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        clear_dir(target)

    print("Reset selesai. Struktur folder tetap ada, isi sudah dibersihkan.")


if __name__ == "__main__":
    main()
