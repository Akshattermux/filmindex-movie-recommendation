"""Fetch and verify the two members of the public Kaggle dataset."""
import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


URL = "https://www.kaggle.com/api/v1/datasets/download/tmdb/tmdb-movie-metadata"
FILES = {
    "tmdb_5000_movies.csv": "1e0584a8dd374120e4ed4f2b81f0d2bd0eb95a554708e4a350d00ed5c46d2825",
    "tmdb_5000_credits.csv": "9d0050599ff88d40366c4841204b1489862bca346bfa46c20b05a65d14508435",
}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(destination):
    destination.mkdir(parents=True, exist_ok=True)
    if all((destination / name).is_file() and file_hash(destination / name) == digest for name, digest in FILES.items()):
        return
    with tempfile.TemporaryDirectory(dir=destination) as scratch:
        tmp = Path(scratch)
        archive = tmp / "dataset.zip"
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError("Install curl, or download the two CSVs from Kaggle into the data directory.")
        subprocess.run(
            [curl, "--fail", "--location", "--silent", "--show-error", "--retry", "2",
             "--connect-timeout", "10", "--max-time", "180", "--max-filesize", "100000000",
             "--output", str(archive), URL],
            check=True,
        )
        with zipfile.ZipFile(archive) as bundle:
            for name, digest in FILES.items():
                if bundle.getinfo(name).file_size > 100_000_000:
                    raise ValueError(f"Unexpected size for {name}")
                with bundle.open(name) as source, (tmp / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                if file_hash(tmp / name) != digest:
                    raise ValueError(f"Kaggle snapshot changed: checksum mismatch for {name}")
        for name in FILES:
            os.replace(tmp / name, destination / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    fetch(args.data_dir)
    print(f"Verified Kaggle movie and credit CSVs in {args.data_dir}")
