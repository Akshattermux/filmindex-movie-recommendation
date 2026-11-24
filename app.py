"""Film Index — Streamlit entry point."""
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from filmindex.model import CatalogError, DEFAULT_MODEL, load_catalog
from filmindex.posters import get_poster_url

ASSETS = Path(__file__).resolve().parent / "assets"
PALETTE = (("#355447", "#d4bb71"), ("#744637", "#f6bb83"), ("#294658", "#9db8cf"),
           ("#5f3849", "#df9ba8"), ("#50542f", "#d4cf89"))

st.set_page_config(page_title="Film Index · Movie recommendations", page_icon="🎞️", layout="wide")


@st.cache_resource(max_entries=2)
def open_release(path, modified, size):
    return load_catalog(path)


@lru_cache(maxsize=2)
def embedded(name, mime):
    return f"data:{mime};base64," + base64.b64encode((ASSETS / name).read_bytes()).decode("ascii")


def poster_token():
    token = os.environ.get("TMDB_READ_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("TMDB_READ_TOKEN", "")
    except StreamlitSecretNotFoundError:
        return ""


def grid(catalog, matches, posters):
    cards = []
    for rank, (movie_id, _) in enumerate(matches):
        movie = catalog.movies[movie_id]
        title = escape(movie["title"])
        genre = escape(next(iter(movie["genres"]), ""))
        meta = " · ".join(filter(None, (escape(movie["year"]), genre)))
        base, ink = PALETTE[rank]
        url = posters.get(movie_id)
        image = f'<img src="{escape(url)}" alt="" loading="lazy">' if url else ""
        cards.append(f"""
        <article class="card"><a href="https://www.themoviedb.org/movie/{movie_id}" target="_blank"
          rel="noopener noreferrer" aria-label="View {title} on TMDB">
          <div class="art form{rank}" aria-hidden="true" style="--base:{base};--ink:{ink}">
            <span class="cover-title">{title}</span>{image}
          </div><p class="meta">{meta}</p><h3>{title}</h3></a></article>""")
    return '<div class="cards">' + "".join(cards) + '</div>'


def run():
    path = Path(os.environ.get("FILMINDEX_MODEL", DEFAULT_MODEL))
    try:
        stat = path.stat()
        catalog = open_release(path, stat.st_mtime_ns, stat.st_size)
    except (OSError, CatalogError):
        st.error("The movie catalog is unavailable. Restore artifacts/catalog.pkl or run python -m filmindex.train.")
        st.stop()

    st.html(ASSETS / "style.css")
    st.html(f'<div class="mast"><div class="brand"><span class="film-mark" aria-hidden="true"></span>Film Index</div>'
            f'<span>{len(catalog.movies):,} films in the catalog</span></div>')
    header = embedded("projection.jpg", "image/jpeg")
    st.html(f"""<section class="hero" style="background-image:linear-gradient(90deg,rgba(9,22,18,.94),
          rgba(9,22,18,.62) 45%,transparent 78%),url({header})">
          <h1>Find your <em>next film.</em></h1><p>Pick a film you love to discover five with similar stories, casts, or filmmakers.</p></section>""")
    initial = catalog.sorted_ids.index(27205) if 27205 in catalog.movies else 0
    seed = st.selectbox("Choose a movie", catalog.sorted_ids, index=initial, format_func=catalog.label)
    matches = catalog.recommend(seed)
    st.html(f'<div class="results"><h2>Five films like <em>{escape(catalog.movies[seed]["title"])}</em></h2></div>')
    deck = st.empty()
    deck.html(grid(catalog, matches, {}))
    with st.expander("Explore the matches"):
        source = catalog.movies[seed]
        for movie_id, score in matches:
            movie = catalog.movies[movie_id]
            st.text(catalog.label(movie_id))
            shared = []
            for key in ("keywords", "directors", "cast", "genres"):
                shared.extend(n for n in movie[key] if n in source[key] and n not in shared)
            detail = "Shared metadata · " + ", ".join(shared[:4]) if shared else "Ranked from the combined content features"
            st.caption(f"{detail}   |   Cosine similarity {score:.3f}")
            if movie["overview"]:
                st.text(movie["overview"])
    logo = embedded("tmdb.svg", "image/svg+xml")
    st.html(f'<footer class="credits"><a href="https://www.themoviedb.org"><img src="{logo}" alt="TMDB"></a>'
            '<span>This product uses the TMDB API but is not endorsed or certified by TMDB.</span></footer>')
    token = poster_token()
    if token:
        ids = [movie_id for movie_id, _ in matches]
        with ThreadPoolExecutor(max_workers=len(ids)) as pool:
            urls = list(pool.map(lambda i: get_poster_url(i, token), ids))
        if any(urls):
            deck.html(grid(catalog, matches, dict(zip(ids, urls))))


if __name__ == "__main__":
    run()
