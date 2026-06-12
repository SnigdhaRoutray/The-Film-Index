"""
app.py  —  Flask REST API for CineRank Movie Search
Endpoints:
  GET  /api/search?q=...&genre=...&year_min=...&year_max=...&n=12
  GET  /api/genres       — list of available genres
  GET  /api/stats        — dataset stats
  GET  /                 — serves index.html
"""

import os
import json
import pickle
import requests
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from pipeline import search, load_pipeline, build_pipeline, TOP_GENRES, DATA_CSV

app = Flask(__name__, static_folder=".")

# ── TMDB API key (optional — graceful fallback if config.py missing) ──────────
TMDB_API_KEY = ""
try:
    from config import TMDB_API_KEY
except ImportError:
    pass

# ── Load or build pipeline on startup ────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
RANKER_PKL  = os.path.join(MODELS_DIR, "ranker.pkl")

print("Initialising pipeline …")
if os.path.exists(RANKER_PKL):
    print("  Loading saved models …")
    df, vec, matrix, title_vec, title_matrix, ranker = load_pipeline()
else:
    print("  No saved models found — building from scratch …")
    df, vec, matrix, title_vec, title_matrix, ranker = build_pipeline(DATA_CSV)
print("  Ready.\n")


def get_movie_poster(title: str, year=None) -> str:
    """Fetch poster URL from TMDB. Returns '' on any failure."""
    if not TMDB_API_KEY:
        return ""
    try:
        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_API_KEY}&query={requests.utils.quote(str(title))}"
        )
        if year:
            url += f"&year={int(year)}"
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        if not data.get("results"):
            return ""
        path = data["results"][0].get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else ""
    except Exception:
        return ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")
@app.route("/Cinema.jpeg")
def hero_image():
    return send_from_directory(".", "Cinema.jpeg")

@app.route("/api/search")
def api_search():
    query    = request.args.get("q", "").strip()
    genre    = request.args.get("genre", "all").strip().lower()
    year_min = request.args.get("year_min", type=int)
    year_max = request.args.get("year_max", type=int)
    n        = min(request.args.get("n", 12, type=int), 30)

    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    results = search(
        query=query,
        df=df,
        vec=vec,
        matrix=matrix,
        title_vec=title_vec,
        title_matrix=title_matrix,
        ranker=ranker,
        top_n=n,
        genre_filter=genre if genre != "all" else None,
        year_min=year_min,
        year_max=year_max,
    )

    if results.empty:
        return jsonify({"results": [], "total": 0})

    output = []
    for _, row in results.iterrows():
        cast_list = [c.strip() for c in str(row["Cast"]).split(",") if c.strip()][:4]
        output.append({
            "title":        row["Title"],
            "genre":        row["Genre"],
            "director":     row["Director"],
            "cast":         cast_list,
            "year":         int(row["Release Year"]),
            "plot_snippet": row["plot_snippet"],
            "wiki_url":     row.get("Wiki Page", ""),
            "score":        round(float(row["score"]), 4),
            "poster_url":   get_movie_poster(row["Title"], row["Release Year"]),
        })

    return jsonify({"results": output, "total": len(output)})


@app.route("/api/genres")
def api_genres():
    genres = ["all"] + sorted(TOP_GENRES)
    return jsonify({"genres": genres})


@app.route("/api/stats")
def api_stats():
    return jsonify({
        "total_movies": int(len(df)),
        "year_min":     int(df["Release Year"].min()),
        "year_max":     int(df["Release Year"].max()),
        "total_genres": int(df["Genre"].nunique()),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
