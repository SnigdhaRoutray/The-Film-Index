# 🎬 The Film Index

A machine learning-powered movie search engine that helps users discover films through movie titles, plot descriptions, genres, cast members, themes, and keywords.

Unlike traditional keyword search, The Film Index combines semantic retrieval with learning-to-rank techniques to surface more relevant results from a dataset of over **34,000 movies**.

---

## Demo Video

https://github.com/user-attachments/assets/78f34222-0ca5-428b-8db7-a497f26a7a0f

---

## Screenshots

### Search Interface

<img width="1891" height="902" alt="Search Interface" src="https://github.com/user-attachments/assets/7d97a174-742c-463c-96cb-74783fd6f064" />

### Search Results

<img width="1917" height="910" alt="Search Results" src="https://github.com/user-attachments/assets/6b4eea92-ba66-416e-abe4-37b3c1ca95e4" />

### Movie Details Modal

<img width="1918" height="911" alt="Movie Details" src="https://github.com/user-attachments/assets/d79327d3-2d34-425e-995d-001d99a64b9b" />

---

## Features

* Search movies by title
* Search using natural-language plot descriptions
* Search by cast members and keywords
* Genre filtering
* Year-range filtering
* Movie posters fetched dynamically from TMDB
* Detailed movie information modal
* Machine learning-based ranking system
* Responsive cinematic user interface

---

## How It Works

The search pipeline combines traditional information retrieval techniques with machine learning ranking.

### 1. Candidate Retrieval

A TF-IDF vectorizer is trained on movie metadata including:

* Title
* Genre
* Cast
* Plot

Cosine similarity is used to retrieve the most relevant candidate movies.

### 2. Feature Engineering

Several ranking features are generated:

| Feature           | Description                            |
| ----------------- | -------------------------------------- |
| TF-IDF Similarity | Query similarity against movie content |
| Title Similarity  | Query similarity against movie title   |
| Exact Title Match | Boost for title word overlap           |
| Keyword Overlap   | Shared words between query and plot    |
| Genre Match       | Genre relevance signal                 |
| Cast Match        | Actor relevance signal                 |
| Year Recency      | Normalized release year feature        |

### 3. Learning-to-Rank

A LightGBM LambdaRank model re-ranks retrieved candidates using engineered features to improve search relevance.

---

## Example Queries

Try searching:

```text
Harry Potter
```

```text
young wizard magic school
```

```text
detective murder mystery
```

```text
space aliens invasion
```

```text
heist bank robbery
```

```text
romantic paris love
```

---

## Tech Stack

### Backend

* Python
* Flask
* Pandas
* Scikit-learn
* LightGBM

### Machine Learning

* TF-IDF Vectorization
* Cosine Similarity Retrieval
* Learning-to-Rank (LambdaRank)

### Frontend

* HTML
* CSS
* JavaScript

### External APIs

* TMDB API (movie posters)

---

## Project Structure

```text
The-Film-Index/
│
├── app.py
├── pipeline.py
├── evaluate.py
├── index.html
├── Cinema.jpeg
├── README.md
│
├── data/
│   └── wiki_movie_plots_deduped.csv
│
└── models/
    ├── vectorizer.pkl
    ├── ranker.pkl
    └── movies.pkl
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/SnigdhaRoutray/The-Film-Index.git
cd The-Film-Index
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure TMDB API Key

Create a file named:

```python
config.py
```

Add:

```python
TMDB_API_KEY = "YOUR_API_KEY"
```

### Run Application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

## Future Improvements

* Transformer-based semantic search
* User watchlists and recommendations
* Personalized ranking
* Hybrid retrieval architecture
* Deployment to a public cloud platform
* Real-time analytics and feedback signals

---

## Dataset

This project uses the Wikipedia Movie Plots dataset containing over 34,000 films.

The dataset is not included in the repository because of its size.

---

## Author

**Snigdha Routray**

AI & Machine Learning Student

Built as a full-stack machine learning project combining information retrieval, learning-to-rank systems, and modern frontend design.


