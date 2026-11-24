# Film Index

A Streamlit movie recommender trained on 4,803 films from Kaggle’s TMDB 5000 snapshot. Choose a movie to see its five closest matches. Open **Explore the matches** to inspect plot summaries, shared metadata and cosine scores. Selecting a different film changes the recommendations immediately. The selected title supplies the preference signal for each request.

## Run the app

The download includes both source CSVs and the trained index. From the extracted `filmindex` folder, use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. On Windows, create the environment with `py -3.12 -m venv .venv` and activate it in PowerShell with `.\.venv\Scripts\Activate.ps1`.

## Enable real posters

Obtain an API Read Access Token from your TMDB account. On the server, set the environment variable `TMDB_READ_TOKEN` or create the following local file. Restart the app after changing credentials. [TMDB API settings](https://www.themoviedb.org/settings/api)

```toml
# .streamlit/secrets.toml
TMDB_READ_TOKEN = "your-read-access-token"
```

The browser receives public image URLs. The server sends the token only to the TMDB movie-details endpoint. The results appear first with typographic covers. When artwork is available, the app replaces the covers with posters. A process caches successful poster paths for 24 hours, limits concurrent requests to five, and respects upstream rate limits. A missing token or a TMDB outage leaves search and rankings available. [TMDB image documentation](https://developer.themoviedb.org/docs/image-basics)

## Where to train

**Kaggle CPU Notebook** is the simplest hosted option. Import `train_on_kaggle.ipynb` into a new notebook, attach **TMDB 5000 Movie Dataset** by **tmdb**, choose a CPU session and run all cells. The notebook discovers both CSVs, verifies the input hashes, trains the model and prints example recommendations. Save the notebook version and download `/kaggle/working/catalog.pkl` from its output. Replace the app’s `artifacts/catalog.pkl` with that file. The notebook includes all training code and uses Kaggle’s scientific Python packages. [Kaggle dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) · [Kaggle notebooks](https://www.kaggle.com/docs/notebooks)

**Local CPU** is the reproducible option. In the project’s activated environment, run the following commands.

```bash
python -m pip install -r requirements-train.txt
python -m filmindex.data
python -m filmindex.train
python -m filmindex.evaluate
python -m unittest discover -s tests -v
```

The fetch command verifies the bundled CSVs. When either file is missing or has changed, it uses curl to download Kaggle’s archive, verifies both SHA-256 hashes and installs the expected files. The original Kaggle files are `data/tmdb_5000_movies.csv` and `data/tmdb_5000_credits.csv`. To experiment on another matching pair, use `python -m filmindex.train --data-dir path/to/csvs --output artifacts/candidate.pkl`. Set `FILMINDEX_MODEL` to an alternate index path to preview it.

## How the ranking works

Pandas joins movies and credits one-to-one by TMDB ID. The pipeline parses the JSON metadata, retains the first five billed cast members and identifies directors by crew job. Scikit-learn fits sparse TF–IDF features independently for plot text, genres, keywords, cast and directors. Multiword names stay together. Plot features include unigrams and bigrams. Titles and release years are display fields.

| Feature | Weight |
| --- | ---: |
| Keywords | 36% |
| Plot overview | 24% |
| Genres | 20% |
| Cast | 12% |
| Directors | 8% |

These are fixed design weights. Each field is L2-normalized and scaled by the square root of its weight before concatenation. Final normalization accounts for missing fields. Batched cosine similarity ranks eligible films, excludes the selected film and resolves score ties by ascending movie ID. Training writes only the five winning IDs and scores per film, together with display metadata, input hashes, feature counts and the training library version. A validated release replaces the previous pickle atomically.

The released model uses 61,801 features and contains 24,015 recommendation entries. Selecting a title reads five stored results. The app never computes an all-pairs similarity matrix during a request. Only deploy catalogs from your own release process. The reader blocks class imports during unpickling and validates the catalog structure.

## Evaluation and limits

The release produced five distinct, ranked, non-self matches for every catalog movie. In a hand-labelled check across 11 franchises, **37 of 38** seed films returned at least one other member of the same franchise in the top five. The included evaluator prints each series’ result. This benchmark measures franchise retrieval. Assess broader discovery quality with viewer feedback on standalone films and cross-franchise matches.

The catalog is a historical snapshot. It has no viewing histories or individual preference labels. People choosing the same title receive the same five films. Sparse descriptions and broad genre overlap can produce weak matches. A cosine score describes content similarity and has no interpretation as a probability that the viewer will like the film.

## Deploy

**Streamlit Community Cloud.** Put the project in a GitHub repository, including `artifacts/catalog.pkl`, `assets/`, `.streamlit/config.toml` and `requirements.txt`. The raw training CSVs can stay local. Create an app with `app.py` as the entry point, select Python 3.12 in Advanced settings, and add `TMDB_READ_TOKEN` to Secrets to enable posters. [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

**Docker.** Build from the project root. If the host already has the token in its environment, Docker forwards it by name.

```bash
docker build -t filmindex:local .
docker run --rm --name filmindex -p 127.0.0.1:8501:8501 -e TMDB_READ_TOKEN filmindex:local
```

The container runs as a non-root user and exposes Streamlit on port 8501. The health check polls `/_stcore/health`. Use the included app test for model and rendering readiness. For an internet-facing deployment, terminate HTTPS at the hosting platform and enable WebSocket forwarding. Keep the prior tagged image available to roll back a release. The GitHub Actions workflow runs tests and builds the image on pushes and pull requests.

TMDB supplies the metadata and movie artwork. Its developer API is available for attributed non-commercial use. For a revenue-generating service, arrange commercial terms with TMDB before launch. The app displays the official TMDB logo and attribution notice. [TMDB API FAQ and commercial licensing](https://developer.themoviedb.org/docs/faq)
