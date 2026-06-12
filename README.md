# CineRank — ML Movie Search

A full ML-powered movie search engine with a cinematic dark-themed frontend.  
Uses **TF-IDF retrieval** + **LightGBM LambdaRank** to re-rank 34 886 movies.

---

## Project Structure

```
movie_search/
├── pipeline.py          ← Feature engineering + model training + search logic
├── app.py               ← Flask REST API
├── index.html           ← Frontend (served by Flask)
├── requirements.txt
├── data/
│   └── wiki_movie_plots_deduped.csv   ← Put the dataset here
└── models/              ← Auto-created; stores trained model artifacts
    ├── movies.pkl
    ├── vectorizer.pkl
    └── ranker.pkl
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Copy `wiki_movie_plots_deduped.csv` into the `data/` folder:
```bash
mkdir -p data
cp /path/to/wiki_movie_plots_deduped.csv data/
```

### 3. Run the app
```bash
python app.py
```

On **first run**, the pipeline will:
1. Load and clean 34 886 movies (~5 seconds)
2. Build a 30 000-term TF-IDF matrix (~6 seconds)
3. Generate synthetic training data (~30 seconds)
4. Train the LightGBM LambdaRank model (~10 seconds)
5. Save everything to `models/`

**Subsequent runs** load the saved models instantly (~3 seconds).

Then open **http://localhost:5000** in your browser.

---

## Features Used by the Ranker

| Feature | Description |
|---|---|
| `tfidf_similarity` | Cosine similarity between query and (title + genre + cast + plot) |
| `title_similarity` | Cosine similarity between query and title only |
| `keyword_overlap` | Raw word overlap count between query and plot |
| `keyword_overlap_r` | Overlap ratio (overlap / query length) |
| `plot_length` | Log-normalised word count of the plot |
| `genre_match` | 1 if any query word appears in the genre string |
| `year_recency` | Normalised release year (newer = higher) |
| `cast_match` | Count of query words found in the cast string |

---

## API Endpoints

| Endpoint | Params | Description |
|---|---|---|
| `GET /api/search` | `q`, `genre`, `year_min`, `year_max`, `n` | Run a search |
| `GET /api/genres` | — | List available genres |
| `GET /api/stats` | — | Dataset statistics |

---

## Files from the Original Project (superseded)

| Old file | Replaced by |
|---|---|
| `search.py` | `pipeline.py` → `search()` function |
| `create_training_data.py` | `pipeline.py` → `build_training_data()` |
| `create_features.py` / `create_genre_features.py` / `build_feature_dataset.py` | `pipeline.py` → `extract_features()` |
| `train_ranker.py` / `train_genre_ranker.py` | `pipeline.py` → `train_ranker()` |
| `generate_genre_training_data.py` / `prepare_genre_training_data.py` | merged into `build_training_data()` |
| `prepare_training_data.py` | debug script — deleted |
