import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._load_and_train()

    def _load_and_train(self):
        """Loads dataset and pre-computes TF-IDF matrix upon server initialization."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset file not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        # Create combined content feature string combining genre and overview
        self.df['combined_features'] = (
            self.df['genre'].fillna('') + " " + self.df['overview'].fillna('')
        )
        
        # Initialize TF-IDF Vectorizer with stop words removal
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        print(f"[Recommender] Model loaded successfully with {len(self.df)} dataset entries.")

    def recommend(self, query, top_n=5):
        """
        Given a text query or title, compute TF-IDF similarity vector and return top_n matching movies.
        Handles both title lookups and natural language text descriptions.
        """
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            return []

        query = query.strip()

        # Check if exact/partial movie title exists in dataset
        matched_movie = self.df[self.df['title'].str.lower() == query.lower()]
        
        if not matched_movie.empty:
            # Title-based recommendation
            idx = matched_movie.index[0]
            sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
            # Sort indices descending by score (exclude self)
            related_indices = sim_scores.argsort()[::-1]
            related_indices = [i for i in related_indices if i != idx][:top_n]
        else:
            # Free-text/Natural language query search recommendation
            query_vec = self.tfidf_vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            related_indices = sim_scores.argsort()[::-1][:top_n]

        results = []
        for idx in related_indices:
            score = float(sim_scores[idx])
            row = self.df.iloc[idx]
            results.append({
                "id": int(row['movie_id']),
                "title": str(row['title']),
                "genre": str(row['genre']),
                "overview": str(row['overview']),
                "rating": float(row['rating']),
                "release_year": int(row['release_year']),
                "similarity_score": round(score * 100, 2)
            })

        return results
