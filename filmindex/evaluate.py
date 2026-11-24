"""Small franchise-retrieval regression benchmark. TMDB IDs are hand-labelled."""
import argparse
from pathlib import Path

from .model import DEFAULT_MODEL, load_catalog

SERIES = {
    "Before": (76, 80, 132344),
    "Alien": (348, 679, 8077, 8078),
    "Bourne": (2501, 2502, 2503, 49040),
    "Ice Age": (425, 950, 8355, 57800),
    "Ocean's": (161, 163, 298),
    "Pirates of the Caribbean": (22, 58, 285, 1865),
    "Shrek": (808, 809, 810, 10192),
    "The Godfather": (238, 240, 242),
    "The Lord of the Rings": (120, 121, 122),
    "The Matrix": (603, 604, 605),
    "Toy Story": (862, 863, 10193),
}


def evaluate(catalog):
    hits = seeds = 0
    for series, members in SERIES.items():
        group_hits = sum(any(other in members for other, _ in catalog.recommend(seed)) for seed in members)
        print(f"{series:24} {group_hits}/{len(members)}")
        hits += group_hits
        seeds += len(members)
    print(f"Franchise hit rate at 5: {hits}/{seeds} ({hits / seeds:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate(load_catalog(parser.parse_args().model))
