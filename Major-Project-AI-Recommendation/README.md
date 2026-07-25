# CineMind AI - Full-Stack AI Recommendation Web Application

[![iNeuBytes VIIP Major Project](https://img.shields.io/badge/iNeuBytes-VIIP%20Major%20Project-indigo.svg)](https://ineubytes.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

A complete full-stack AI web application that provides intelligent movie and content recommendations using Natural Language Processing (NLP) and Machine Learning techniques.

Developed for the **iNeuBytes Virtual Internship Program (VIIP) - Artificial Intelligence (Course ID: AIINB20726)**.

---

## 🌟 Features

- **Option A Recommendation Engine**: Content-based filtering using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization and **Cosine Similarity** scoring.
- **Model Efficiency**: Model pre-computes TF-IDF feature matrices **once at server startup**, guaranteeing fast sub-100ms recommendation response times.
- **Robust Flask Backend**:
  - `/health`: Live status verification endpoint.
  - `/recommend`: REST API accepting both POST JSON payloads and GET query parameters.
  - **Graceful Error Handling**: Input validation ensuring empty or malformed requests return structured 400 Bad Request payloads instead of server crashes.
- **Modern Glassmorphic UI**: High-performance, responsive frontend with live match percentage badges, response time counters, and quick suggestion chips.
- **Postman Test Suite**: Included `postman/api_collection.json` containing test cases for success, edge, and error scenarios.

---

## 📁 Repository Structure

```
ai-recommendation-app/
├── backend/
│   ├── app.py                 # Main Flask server with CORS & endpoints (/health, /recommend)
│   ├── recommender.py         # ML Recommender class (TF-IDF + Cosine Similarity)
│   ├── dataset/
│   │   └── movies.csv         # Curated movie dataset (genres, overviews, ratings)
│   └── requirements.txt       # Production Python dependencies
├── frontend/
│   ├── index.html             # Glassmorphic HTML5 UI
│   ├── style.css              # Custom styling & dark-theme design system
│   └── app.js                 # Frontend API consumer & UI state manager
├── postman/
│   └── api_collection.json    # Postman test collection
├── README.md                  # Comprehensive project documentation
└── .gitignore                 # Excludes environment & build artifacts
```

---

## 🤖 Model & Architecture Explanation

### 1. Dataset
The recommendation model operates on a curated dataset (`backend/dataset/movies.csv`) containing detailed attributes:
- `title`: Movie name
- `genre`: Categories (Sci-Fi, Drama, Action, Comedy, etc.)
- `overview`: Full narrative synopsis
- `rating` & `release_year`: Metadata fields displayed on recommendation cards.

### 2. Machine Learning Algorithm
1. **Feature Extraction**: Concatenates `genre` and `overview` text per entry to form rich descriptive document strings.
2. **TF-IDF Vectorization**: `TfidfVectorizer(stop_words='english')` converts textual descriptions into high-dimensional numerical feature vectors.
3. **Similarity Calculation**: Computes pairwise **Cosine Similarity** ($\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$) between the query vector and all dataset vectors.
4. **Ranking**: Returns the top $N$ candidates sorted by similarity percentage score.

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- Python 3.9+
- Git

### 1. Clone & Setup Backend
```bash
cd backend
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Flask Server
```bash
python app.py
```
*Server runs at `http://127.0.0.1:5000`*

### 3. Launch Frontend
Open `frontend/index.html` in your web browser (or serve via Live Server / Python http.server).

---

## 🧪 API Endpoints & Testing

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response (200 OK)**:
```json
{
  "status": "OK",
  "message": "AI Recommendation Service is running healthy.",
  "dataset_loaded": 20
}
```

### 2. Get Recommendations
- **Endpoint**: `POST /recommend`
- **Header**: `Content-Type: application/json`
- **Body**:
```json
{
  "query": "Inception",
  "top_n": 5
}
```
- **Response (200 OK)**:
```json
{
  "status": "success",
  "query": "Inception",
  "count": 5,
  "response_time_ms": 12.4,
  "recommendations": [
    {
      "id": 4,
      "title": "The Matrix",
      "genre": "Sci-Fi Action",
      "rating": 8.7,
      "release_year": 1999,
      "similarity_score": 45.18
    }
  ]
}
```

### 3. Postman Collection
Import `postman/api_collection.json` into Postman to run automated API verification tests.

---

## 🌐 Deployment Guide (Render / Railway / PythonAnywhere)

### Deployment to Render
1. Push this code repository to GitHub.
2. Log into [Render.com](https://render.com) and click **New > Web Service**.
3. Connect your GitHub repository.
4. Set Build & Start settings:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Create Web Service**. Your Flask AI backend will be live at a public URL!

---

## 📄 License
Created as part of the iNeuBytes VIIP Major Project.
