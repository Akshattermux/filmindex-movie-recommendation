import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import requests

from filmindex import posters


def response(status=200, path="/art.jpg", headers=None):
    result = MagicMock(status_code=status, headers=headers or {})
    result.__enter__.return_value = result
    result.json.return_value = {"poster_path": path}
    return result


class PosterTests(unittest.TestCase):
    def setUp(self):
        self.cache = posters._PosterCache()
        self.get = self.enterContext(patch.object(posters.requests, "get"))

    def test_token_stays_on_server_and_warm_hit_uses_no_network(self):
        self.get.return_value = response()
        with patch.object(posters, "_cache", self.cache):
            self.assertIsNone(posters.get_poster_url(42, " "))
            url = posters.get_poster_url(42, "  token  ")
            self.assertEqual(url, "https://image.tmdb.org/t/p/w500/art.jpg")
            self.assertEqual(posters.get_poster_url(42, "token"), url)
        self.get.assert_called_once()
        args, kwargs = self.get.call_args
        self.assertEqual(args, ("https://api.themoviedb.org/3/movie/42",))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer token")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertTrue(all(0 < value <= 3 for value in kwargs["timeout"]))

    def test_null_malformed_and_404_are_cached(self):
        for movie_id, (status, path) in enumerate(
            [(200, None), (200, "//example.com/a.jpg"), (404, "/art.jpg")], 1
        ):
            with self.subTest(movie_id=movie_id):
                self.get.reset_mock()
                self.get.return_value = response(status, path)
                for _ in range(2):
                    self.assertIsNone(self.cache.get(movie_id, "token"))
                self.get.assert_called_once()

    def test_limit_respects_retry_after_across_movie_ids(self):
        self.get.side_effect = [response(429, headers={"Retry-After": "45"}), response()]
        with patch.object(posters.time, "monotonic", return_value=10) as now:
            self.assertIsNone(self.cache.get(1, "token"))
            now.return_value = 54
            self.assertIsNone(self.cache.get(2, "token"))
            now.return_value = 55
            self.assertIsNotNone(self.cache.get(2, "token"))
        self.assertEqual(self.get.call_count, 2)

    def test_replacing_invalid_token_recovers_immediately(self):
        self.get.side_effect = [response(401), response()]
        self.assertIsNone(self.cache.get(1, "old"))
        self.assertIsNone(self.cache.get(2, "old"))
        self.assertIsNotNone(self.cache.get(1, "new"))
        self.assertEqual(self.get.call_count, 2)

    def test_temporarily_failed_request_recovers(self):
        self.get.side_effect = [requests.Timeout(), response()]
        with patch.object(posters.time, "monotonic", return_value=100) as now:
            self.assertIsNone(self.cache.get(42, "token"))
            self.assertIsNone(self.cache.get(42, "token"))
            now.return_value = 131
            self.assertIsNotNone(self.cache.get(42, "token"))
        self.assertEqual(self.get.call_count, 2)

    def test_bounds_parallel_requests_and_skips_duplicate_in_flight(self):
        started = threading.Barrier(6)
        release = threading.Event()

        def slow(*args, **kwargs):
            started.wait(timeout=5)
            release.wait(timeout=5)
            return response()

        self.get.side_effect = slow
        with ThreadPoolExecutor(max_workers=5) as pool:
            pending = [pool.submit(self.cache.get, i, "token") for i in range(1, 6)]
            try:
                started.wait(timeout=5)
                self.assertIsNone(self.cache.get(1, "token"))
                self.assertIsNone(self.cache.get(6, "token"))
            finally:
                release.set()
            self.assertTrue(all(task.result() for task in pending))
        self.assertEqual(self.get.call_count, 5)
