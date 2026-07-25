import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import MovieRecommender

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend access

# Load model ONCE when server starts
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'movies.csv')
recommender = MovieRecommender(DATASET_PATH)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint required by VIIP spec."""
    return jsonify({
        "status": "OK",
        "message": "AI Recommendation Service is running healthy.",
        "dataset_loaded": len(recommender.df) if recommender.df is not None else 0
    }), 200

@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    """
    Recommendation endpoint required by VIIP spec.
    Accepts JSON body: {"query": "Inception", "top_n": 5}
    or URL parameter: /recommend?query=Inception&top_n=5
    """
    start_time = time.time()
    
    try:
        # Extract query and top_n parameter from POST JSON or GET query params
        if request.method == 'POST':
            data = request.get_json(silent=True) or {}
            query = data.get('query', '')
            top_n = data.get('top_n', 5)
        else:
            query = request.args.get('query', '')
            top_n = request.args.get('top_n', 5)

        # Validate input
        if not query or not str(query).strip():
            return jsonify({
                "error": "Bad Request",
                "message": "Please provide a valid 'query' parameter (e.g. movie title or genre description)."
            }), 400

        try:
            top_n = int(top_n)
            top_n = max(1, min(top_n, 10))  # limit between 1 and 10
        except ValueError:
            top_n = 5

        # Get recommendations from ML model
        results = recommender.recommend(query=str(query), top_n=top_n)
        
        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "status": "success",
            "query": str(query),
            "count": len(results),
            "response_time_ms": execution_time_ms,
            "recommendations": results
        }), 200

    except Exception as e:
        # Prevent server crashing and return 500 error payload
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
