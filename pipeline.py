"""
pipeline.py
-----------
Builds and saves:
  - TF-IDF vectorizer  (models/vectorizer.pkl)
  - LightGBM ranker    (models/ranker.pkl)
  - Processed movie DF (data/movies.pkl)

Features used (v2):
  1. tfidf_similarity   — cosine sim query vs plot+title+genre (ngrams)
  2. title_similarity   — cosine sim query vs title only
  3. title_exact_boost  — fraction of query words present in title
  4. keyword_overlap    — raw word overlap count (query ∩ plot)
  5. keyword_overlap_r  — overlap ratio (overlap / query_len)
  6. genre_match        — 1 if query word appears in genre string
  7. year_recency       — normalised release year (newer = higher)
  8. cast_match         — count of query words found in cast string
"""

import os, pickle, random
from evaluate import evaluate_ranker
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightgbm import LGBMRanker

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV   = os.path.join(os.path.dirname(__file__), "data", "wiki_movie_plots_deduped.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MOVIES_PKL = os.path.join(MODELS_DIR, "movies.pkl")
VEC_PKL    = os.path.join(MODELS_DIR, "vectorizer.pkl")
RANKER_PKL = os.path.join(MODELS_DIR, "ranker.pkl")

FEATURE_COLS = [
    "tfidf_similarity",
    "title_similarity",
    "title_exact_boost",
    "keyword_overlap",
    "keyword_overlap_r",
    "genre_match",
    "year_recency",
    "cast_match",
]

# ── Genre normalisation ────────────────────────────────────────────────────────
GENRE_MAP = {
    "sci-fi": "science fiction",
    "romantic comedy": "romance",
    "crime drama": "crime",
    "comedy, drama": "drama",
    "comedy drama": "drama",
    "action/adventure": "adventure",
    "animated": "animation",
}

TOP_GENRES = [
    "drama", "comedy", "romance", "horror", "action",
    "crime", "thriller", "western", "science fiction",
    "adventure", "musical", "mystery", "animation", "war",
]

QUERY_EXPANSION = {
    "spaceship": ["spacecraft","rocket","starship"],
    "alien":     ["extraterrestrial","creature"],
    "love":      ["romance","relationship"],
    "war":       ["battle","army","military"],
    "detective": ["investigator","sleuth"],
    "wizard":    ["magic","sorcerer","witch","spell"],
    "vampire":   ["bloodsucker","undead","dracula"],
    "zombie":    ["undead","living dead"],
    "robot":     ["android","cyborg","machine"],
}


# ── 1. Load & clean ────────────────────────────────────────────────────────────
def load_movies(csv_path: str) -> pd.DataFrame:
    print("Loading dataset …")
    df = pd.read_csv(csv_path)
    df["Plot"]         = df["Plot"].fillna("")
    df["Cast"]         = df["Cast"].fillna("")
    df["Genre"]        = df["Genre"].fillna("unknown").str.lower().str.strip()
    df["Genre"]        = df["Genre"].replace(GENRE_MAP)
    df["Director"]     = df["Director"].fillna("unknown")
    df["Release Year"] = pd.to_numeric(df["Release Year"], errors="coerce").fillna(1980)
    # Title boosted 4x in search_text
    df["search_text"] = (
        df["Title"] + " " + df["Title"] + " " + df["Title"] + " " + df["Title"] + " "
        + df["Genre"] + " " + df["Genre"] + " "
        + df["Cast"] + " "
        + df["Plot"]
    )
    print(f"  {len(df):,} movies loaded.")
    return df


# ── 2. Build TF-IDF ───────────────────────────────────────────────────────────
def build_vectorizer(df: pd.DataFrame):
    print("Building TF-IDF vectorizer …")
    vec = TfidfVectorizer(
        stop_words="english", max_features=30000,
        sublinear_tf=True, min_df=2, ngram_range=(1, 2),
    )
    matrix = vec.fit_transform(df["search_text"])
    title_vec = TfidfVectorizer(
        stop_words="english", max_features=10000,
        sublinear_tf=True, ngram_range=(1, 2),
    )
    title_matrix = title_vec.fit_transform(df["Title"].fillna(""))
    print(f"  Vocab size: {len(vec.vocabulary_):,}")
    return vec, matrix, title_vec, title_matrix


# ── 3. Feature extraction ──────────────────────────────────────────────────────
def extract_features(query, df, vec, matrix, title_vec, title_matrix, indices=None):
    if indices is None:
        indices = list(range(len(df)))
    sub_df     = df.iloc[indices].reset_index(drop=True)
    sub_matrix = matrix[indices]
    sub_title  = title_matrix[indices]
    q_vec       = vec.transform([query])
    q_title_vec = title_vec.transform([query])
    tfidf_sim   = cosine_similarity(q_vec, sub_matrix)[0]
    title_sim   = cosine_similarity(q_title_vec, sub_title)[0]
    q_words = set(query.lower().split())

    def _title_exact(title):
        tw = set(str(title).lower().split())
        return len(q_words & tw) / max(len(q_words), 1)

    title_exact  = sub_df["Title"].apply(_title_exact).values.astype(float)

    def _overlap(plot):
        pw = set(str(plot).lower().split())
        ov = len(q_words & pw)
        return ov, ov / max(len(q_words), 1)

    overlaps     = sub_df["Plot"].apply(_overlap)
    kw_overlap   = overlaps.apply(lambda x: x[0]).values.astype(float)
    kw_overlap_r = overlaps.apply(lambda x: x[1]).values.astype(float)
    plot_len     = np.log1p(sub_df["Plot"].str.split().str.len().fillna(0))
    genre_match  = sub_df["Genre"].apply(
                    lambda g: float(any(w in g for w in q_words))).values
    yr_norm      = np.clip((sub_df["Release Year"].values.astype(float) - 1900) / 120, 0, 1)
    cast_match   = sub_df["Cast"].apply(
                    lambda c: float(sum(w in str(c).lower() for w in q_words))).values

    feats = pd.DataFrame({
        "tfidf_similarity":  tfidf_sim,
        "title_similarity":  title_sim,
        "title_exact_boost": title_exact,
        "keyword_overlap":   kw_overlap,
        "keyword_overlap_r": kw_overlap_r,
        "plot_length":       plot_len.values,
        "genre_match":       genre_match,
        "year_recency":      yr_norm,
        "cast_match":        cast_match,
    })
    return feats, sub_df


# ── 4. Training data ───────────────────────────────────────────────────────────
def build_training_data(df, vec, matrix, title_vec, title_matrix):
    print("Generating training data …")
    hand_queries = [
        ("harry potter",                      "fantasy"),
        ("harry potter wizard school",         "fantasy"),
        ("lord of the rings elf hobbit",       "adventure"),
        ("star wars jedi force galaxy",        "science fiction"),
        ("the godfather mafia crime family",   "crime"),
        ("young wizard magic school",          "science fiction"),
        ("detective murder mystery investigation","crime"),
        ("space aliens invasion earth",        "science fiction"),
        ("romantic love story couple",         "romance"),
        ("superhero fights villain city",      "action"),
        ("ghost haunted house horror",         "horror"),
        ("cowboy outlaw sheriff western",      "western"),
        ("war soldiers battle battlefield",    "war"),
        ("comedy funny misunderstanding",      "comedy"),
        ("heist bank robbery escape",          "crime"),
        ("monster creature attacks village",   "horror"),
        ("time travel adventure future",       "science fiction"),
        ("family drama relationships secrets", "drama"),
        ("spy secret agent mission",           "thriller"),
        ("serial killer police chase",         "thriller"),
        ("singing dancing musical stage",      "musical"),
        ("animated cartoon adventure",         "animation"),
        ("pirate ship treasure island",        "adventure"),
        ("vampire supernatural romance",       "horror"),
        ("teenage high school coming of age",  "drama"),
        ("alien planet crew exploration",      "science fiction"),
        ("martial arts kung fu fight",         "action"),
        ("hitman assassin contract",           "crime"),
        ("disaster survival earthquake",       "action"),
        ("scientist experiment discovery",     "science fiction"),
        ("princess fairy tale kingdom",        "fantasy"),
        ("dinosaur prehistoric park",          "science fiction"),
        ("submarine navy underwater",          "war"),
        ("amnesia identity mystery",           "thriller"),
        ("batman dark knight gotham",          "action"),
        ("inception dream heist",              "science fiction"),
    ]
    all_rows, all_groups = [], []
    for query, _ in hand_queries:
        feats, sub = extract_features(query, df, vec, matrix, title_vec, title_matrix)
        scores = (feats["tfidf_similarity"].values
                  + 2.0 * feats["title_similarity"].values
                  + 3.0 * feats["title_exact_boost"].values)
        si = np.argsort(scores)[::-1]
        pos  = si[:3]
        mid  = np.random.choice(si[20:60], size=min(4,40), replace=False)
        neg  = np.random.choice(si[-200:], size=5, replace=False)
        rows = []
        for i in pos:
            r = feats.iloc[i].to_dict(); r["relevance"] = 3; rows.append(r)
        for i in mid:
            r = feats.iloc[i].to_dict(); r["relevance"] = 1; rows.append(r)
        for i in neg:
            r = feats.iloc[i].to_dict(); r["relevance"] = 0; rows.append(r)
        all_rows.extend(rows); all_groups.append(len(rows))

    for genre in TOP_GENRES:
        feats, _ = extract_features(genre, df, vec, matrix, title_vec, title_matrix)
        gm = df[df["Genre"]==genre].index.tolist()
        om = df[df["Genre"]!=genre].index.tolist()
        if len(gm) < 3: continue
        rows = []
        for i in random.sample(gm, min(5,len(gm))):
            r = feats.iloc[i].to_dict(); r["relevance"] = 3; rows.append(r)
        for i in random.sample(om, min(5,len(om))):
            r = feats.iloc[i].to_dict(); r["relevance"] = 0; rows.append(r)
        all_rows.extend(rows); all_groups.append(len(rows))

    train_df = pd.DataFrame(all_rows)
    print(f"  Training rows: {len(train_df):,}, groups: {len(all_groups)}")
    return train_df, all_groups


# ── 5. Train ranker ────────────────────────────────────────────────────────────
def train_ranker(train_df, groups):
    print("Training LightGBM ranker …")
    X = train_df[FEATURE_COLS]
    y = train_df["relevance"]
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=3,
        random_state=42, verbose=-1,
    )
    model.fit(X, y, group=groups)
    print("  Feature importances:")
    imp_dict = {}
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x:-x[1]):
        print(f"    {feat:<22} {imp}")
        imp_dict[feat] = int(imp)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR,"feature_importance.pkl"), "wb") as f:
        pickle.dump(imp_dict, f)
    return model


# ── 6. Search ──────────────────────────────────────────────────────────────────
def search(query, df, vec, matrix, title_vec, title_matrix, ranker,
           top_n=12, genre_filter=None, year_min=None, year_max=None):
    """ML-ranked movie search with improved title-aware candidate selection."""

    # Query expansion
    expanded = query
    for word in query.lower().split():
        if word in QUERY_EXPANSION:
            expanded += " " + " ".join(QUERY_EXPANSION[word])
    query = expanded

    # Pre-filter
    mask = pd.Series([True] * len(df))
    if genre_filter and genre_filter != "all":
        mask &= df["Genre"].str.contains(genre_filter, case=False, na=False)
    if year_min:
        mask &= df["Release Year"] >= year_min
    if year_max:
        mask &= df["Release Year"] <= year_max
    indices = df[mask].index.tolist()
    if not indices:
        return pd.DataFrame()

    # Candidate retrieval — title-first scoring
    q_vec       = vec.transform([query])
    q_title_vec = title_vec.transform([query])
    raw_s   = cosine_similarity(q_vec,       matrix[indices])[0]
    title_s = cosine_similarity(q_title_vec, title_matrix[indices])[0]

    q_words = set(query.lower().split())
    title_exact = np.array([
        len(q_words & set(str(df.iloc[idx]["Title"]).lower().split())) / max(len(q_words), 1)
        for idx in indices
    ])
    combined = 0.35 * raw_s + 0.65 * title_s + 2.0 * title_exact

    top500   = np.argsort(combined)[::-1][:500]
    cand_idx = [indices[i] for i in top500]

    # Feature extraction + ML re-ranking
    feats, sub_df = extract_features(query, df, vec, matrix, title_vec, title_matrix, cand_idx)

    # Check if ranker has the new feature set; fall back to formula if not
    ranker_features = getattr(ranker, "feature_name_", [])
    if "title_exact_boost" in ranker_features:
        scores = ranker.predict(feats[FEATURE_COLS])
    else:
        # Old ranker — use weighted formula directly (much better than broken ranker)
        scores = (
            0.35 * feats["tfidf_similarity"].values
            + 0.65 * feats["title_similarity"].values
            + 2.0  * feats["title_exact_boost"].values
            + 0.1  * feats["keyword_overlap_r"].values
            + 0.05 * feats["genre_match"].values
            + 0.03 * feats["year_recency"].values
        )

    ranked  = np.argsort(scores)[::-1][:top_n]
    sub_df["tfidf_similarity"] = feats["tfidf_similarity"]
    sub_df["title_similarity"] = feats["title_similarity"]
    sub_df["genre_match"]      = feats["genre_match"]
    results = sub_df.iloc[ranked][[
        "Title","Genre","Director","Cast","Release Year","Plot","Wiki Page",
        "tfidf_similarity","title_similarity","genre_match",
    ]].copy()
    results["score"]        = scores[ranked]
    results["plot_snippet"] = results["Plot"].str[:300]
    return results.reset_index(drop=True)


# ── 7. Build & save ────────────────────────────────────────────────────────────
def build_pipeline(csv_path=DATA_CSV):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df                                   = load_movies(csv_path)
    vec, matrix, title_vec, title_matrix = build_vectorizer(df)
    train_df, groups                     = build_training_data(df, vec, matrix, title_vec, title_matrix)
    ranker                               = train_ranker(train_df, groups)
    evaluate_ranker(ranker, train_df[FEATURE_COLS], train_df["relevance"])
    print("Saving models …")
    with open(MOVIES_PKL, "wb") as f: pickle.dump({"df":df,"matrix":matrix,"title_matrix":title_matrix}, f)
    with open(VEC_PKL,    "wb") as f: pickle.dump({"vec":vec,"title_vec":title_vec}, f)
    with open(RANKER_PKL, "wb") as f: pickle.dump(ranker, f)
    print("Pipeline built successfully.")
    return df, vec, matrix, title_vec, title_matrix, ranker


def load_pipeline():
    with open(MOVIES_PKL, "rb") as f: d = pickle.load(f)
    with open(VEC_PKL,    "rb") as f: v = pickle.load(f)
    with open(RANKER_PKL, "rb") as f: ranker = pickle.load(f)
    return d["df"], v["vec"], d["matrix"], v["title_vec"], d["title_matrix"], ranker


if __name__ == "__main__":
    build_pipeline()
