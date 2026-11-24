"""Portable read-only serving index. The pickle contains only Python primitives."""
import math
import pickle
from pathlib import Path

TOP_K = 5
SCHEMA = 1
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "catalog.pkl"


class CatalogError(ValueError):
    """The deployed release is missing, damaged, or unsupported."""


class _CatalogReader(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Catalogs cannot import Python objects")


class Catalog:
    def __init__(self, release):
        if release["schema"] != SCHEMA:
            raise CatalogError("Unsupported catalog version")
        self.movies = release["movies"]
        self.neighbors = release["neighbors"]
        self.metadata = release["metadata"]
        if len(self.movies) <= TOP_K or self.movies.keys() != self.neighbors.keys():
            raise CatalogError("Incomplete catalog")
        for movie_id, movie in self.movies.items():
            if not isinstance(movie_id, int) or not movie["title"]:
                raise CatalogError("Invalid movie record")
            for key in ("genres", "directors", "keywords", "cast", "overview", "year", "title"):
                if key not in movie:
                    raise CatalogError("Incomplete movie record")
            matches = self.neighbors[movie_id]
            ids = {other for other, _ in matches}
            if len(matches) != TOP_K or len(ids) != TOP_K or movie_id in ids or not ids <= self.movies.keys():
                raise CatalogError("Invalid recommendation IDs")
            scores = [score for _, score in matches]
            if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores) or scores != sorted(scores, reverse=True):
                raise CatalogError("Invalid cosine scores")
        self.sorted_ids = sorted(self.movies, key=lambda i: (self.movies[i]["title"].casefold(), self.movies[i]["year"], i))

    def label(self, movie_id):
        movie = self.movies[movie_id]
        return f"{movie['title']} ({movie['year'] or 'year unknown'})"

    def recommend(self, movie_id):
        return self.neighbors[movie_id]


def load_catalog(path=DEFAULT_MODEL):
    try:
        with Path(path).open("rb") as source:
            release = _CatalogReader(source).load()
            if source.read(1):
                raise CatalogError("Unexpected trailing bytes")
        return Catalog(release)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise CatalogError(f"Cannot load catalog from {path}. Restore it or rerun training.") from exc
