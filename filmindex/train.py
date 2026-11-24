"""Train a content-based top-five cosine index on the TMDB 5000 CSV files."""
import argparse
import json
import os
import pickle
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .data import DEFAULT_DATA, FILES, file_hash
from .model import Catalog, DEFAULT_MODEL, SCHEMA, TOP_K

# Each independently normalized block contributes this fraction of the dot product
# when both films have data in every block.
WEIGHTS = {"overview": 0.24, "genres": 0.20, "keywords": 0.36, "cast": 0.12, "directors": 0.08}


def names(raw, field, movie_id, *, job=None, limit=None):
    if not raw:
        return []
    try:
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError("expected a list")
        if job:
            entries = [e for e in entries if e.get("job") == job]
        if field == "cast":
            entries.sort(key=lambda e: e.get("order", len(entries)))
        cleaned = [e["name"].strip() for e in entries if e.get("name")]
        return list(dict.fromkeys(cleaned))[:limit]
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"Invalid {field} metadata for TMDB ID {movie_id}") from exc


def read_movies(directory):
    films = pd.read_csv(directory / "tmdb_5000_movies.csv", keep_default_na=False,
                        usecols=["id", "title", "release_date", "overview", "genres", "keywords"])
    credits = pd.read_csv(directory / "tmdb_5000_credits.csv", keep_default_na=False,
                          usecols=["movie_id", "cast", "crew"]).rename(columns={"movie_id": "id"})
    merged = films.merge(credits, on="id", how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Movie and credits IDs differ. Use a matching dataset pair.")
    movies = {}
    for row in merged.sort_values("id").itertuples(index=False):
        if not row.title.strip() or row.id <= 0:
            raise ValueError(f"Invalid movie identity: {row.id}")
        movies[int(row.id)] = {
            "title": row.title.strip(), "year": row.release_date[:4], "overview": row.overview.strip(),
            "genres": names(row.genres, "genres", row.id),
            "keywords": names(row.keywords, "keywords", row.id),
            "cast": names(row.cast, "cast", row.id, limit=5),
            "directors": names(row.crew, "crew", row.id, job="Director"),
        }
    return movies


def identity(tokens):
    return tokens


def features(movies):
    blocks = []
    for field, weight in WEIGHTS.items():
        if field == "overview":
            corpus = [m[field] for m in movies.values()]
            vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, strip_accents="unicode",
                                         dtype=np.float32, max_features=40000, ngram_range=(1, 2))
        else:
            corpus = [[re.sub(r"\s+", " ", term.casefold()) for term in m[field]] for m in movies.values()]
            vectorizer = TfidfVectorizer(analyzer=identity, dtype=np.float32)
        try:
            block = vectorizer.fit_transform(corpus)
        except ValueError as exc:
            if "empty vocabulary" in str(exc):
                continue
            raise
        blocks.append(block.multiply(np.float32(weight ** 0.5)))
    if not blocks:
        raise ValueError("No usable content features")
    matrix = normalize(hstack(blocks, format="csr"))
    if np.any(matrix.getnnz(axis=1) == 0):
        raise ValueError("Some movies have no usable content. Check the input metadata.")
    return matrix


def nearest(movies, matrix, batch_size=128):
    ids = np.array(list(movies))
    if len(ids) <= TOP_K:
        raise ValueError("At least six movies are required")
    result = {}
    for start in range(0, len(ids), batch_size):
        scores = cosine_similarity(matrix[start:start + batch_size], matrix)
        for offset, row in enumerate(scores):
            index = start + offset
            row[index] = -np.inf
            # IDs are ascending, providing a stable tie break at the top-five boundary.
            best = np.argsort(-row, kind="stable")[:TOP_K]
            result[int(ids[index])] = [(int(ids[j]), float(np.clip(row[j], 0, 1))) for j in best]
    return result


def train(data_dir, output):
    hashes = {name: file_hash(data_dir / name) for name in FILES}
    movies = read_movies(data_dir)
    matrix = features(movies)
    release = {"schema": SCHEMA, "movies": movies, "neighbors": nearest(movies, matrix), "metadata": {
        "weights": WEIGHTS, "input_sha256": hashes, "sklearn": sklearn.__version__,
        "movie_count": len(movies), "feature_count": matrix.shape[1],
    }}
    Catalog(release)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp:
        path = Path(temp.name)
        try:
            pickle.dump(release, temp, protocol=5)
            temp.flush()
            os.fsync(temp.fileno())
            temp.close()
            os.replace(path, output)
        finally:
            path.unlink(missing_ok=True)
    print(f"Saved {len(movies):,} films and {matrix.shape[1]:,} features to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    train(args.data_dir, args.output)
