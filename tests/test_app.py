import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from filmindex.model import load_catalog


class AppTests(unittest.TestCase):
    def test_selecting_a_film_updates_all_five_cards_without_network(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with patch.dict(os.environ, {"TMDB_READ_TOKEN": ""}), patch("requests.get", side_effect=AssertionError):
            app = AppTest.from_file(str(app_path), default_timeout=20).run()
            self.assertFalse(app.exception)
            app.selectbox[0].select(862).run()
            self.assertFalse(app.exception)
        deck = next(node.proto.body for node in app.get("html") if 'class="cards"' in node.proto.body)
        self.assertEqual(deck.count('class="card"'), 5)
        for movie_id, score in load_catalog().recommend(862):
            self.assertIn(f"https://www.themoviedb.org/movie/{movie_id}\"", deck)
        self.assertNotIn("https://www.themoviedb.org/movie/862\"", deck)
