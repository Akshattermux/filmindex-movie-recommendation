import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from filmindex.model import CatalogError, load_catalog
from filmindex.train import nearest, train


class RankingTests(unittest.TestCase):
    def test_cosine_order_and_tie_cutoff(self):
        matrix = csr_matrix([[1, 0], [0, 2], [3, 0], [1, 1], [1, 2], [0, 3], [0, 1]], dtype=float)
        matches = nearest(dict.fromkeys(range(1, 8)), matrix, batch_size=2)[1]
        self.assertEqual([i for i, _ in matches], [3, 4, 5, 2, 6])
        np.testing.assert_allclose([s for _, s in matches], [1, 1 / np.sqrt(2), 1 / np.sqrt(5), 0, 0])

    def test_release_rejects_truncated_or_nonprimitive_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "catalog.pkl"
            for content in (b"broken", pickle.dumps(Path("example"))):
                path.write_bytes(content)
                with self.assertRaises(CatalogError):
                    load_catalog(path)

    def test_training_joins_on_id_and_preserves_previous_release_on_error(self):
        films = [dict(id=i, title="Remake", release_date=f"200{i}-01-01", overview=f"Escape through space {i}",
                      genres='[{"name":"Adventure"}]', keywords="[]") for i in range(1, 8)]
        credits = [dict(movie_id=i, cast=f'[{{"name":"Actor {i}"}}]', crew="[]") for i in reversed(range(1, 8))]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pd.DataFrame(films).to_csv(root / "tmdb_5000_movies.csv", index=False)
            credit_path = root / "tmdb_5000_credits.csv"
            pd.DataFrame(credits).to_csv(credit_path, index=False)
            artifact = root / "catalog.pkl"
            train(root, artifact)
            catalog = load_catalog(artifact)
            self.assertEqual(catalog.movies[1]["cast"], ["Actor 1"])
            self.assertEqual(catalog.label(2), "Remake (2002)")
            previous = artifact.read_bytes()
            pd.DataFrame(credits[:-1]).to_csv(credit_path, index=False)
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                train(root, artifact)
            self.assertEqual(previous, artifact.read_bytes())
