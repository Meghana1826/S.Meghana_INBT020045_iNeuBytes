// Base URL for Flask backend API (Default local 5000)
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://127.0.0.1:5000' 
    : '';

document.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    setupEventListeners();
});

// Check health of backend server on startup
async function checkBackendHealth() {
    const statusText = document.getElementById('statusText');
    const statusDot = document.querySelector('.status-pill .dot');

    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'OK') {
            statusText.textContent = `Backend Online (${data.dataset_loaded} movies)`;
            statusDot.classList.add('online');
            statusDot.classList.remove('offline');
        } else {
            throw new Error("Server unhealthy");
        }
    } catch (err) {
        console.warn("Backend connection failed:", err);
        statusText.textContent = "Backend Disconnected";
        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
    }
}

function setupEventListeners() {
    const form = document.getElementById('recommendForm');
    const tagChips = document.querySelectorAll('.tag-chip');
    const queryInput = document.getElementById('queryInput');

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (query) {
            fetchRecommendations(query);
        }
    });

    tagChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            queryInput.value = query;
            fetchRecommendations(query);
        });
    });
}

// Fetch recommendations from Flask backend API
async function fetchRecommendations(query) {
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');
    const cardsGrid = document.getElementById('cardsGrid');
    const resultCount = document.getElementById('resultCount');
    const responseTime = document.getElementById('responseTime');

    // UI state transitions
    loadingState.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorBanner.classList.add('hidden');

    try {
        const startTime = performance.now();
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query, top_n: 6 })
        });

        const data = await response.json();
        const duration = Math.round(performance.now() - startTime);

        loadingState.classList.add('hidden');

        if (!response.ok) {
            throw new Error(data.message || 'Failed to fetch recommendations.');
        }

        if (!data.recommendations || data.recommendations.length === 0) {
            errorMessage.textContent = `No relevant recommendations found for "${query}". Try another query!`;
            errorBanner.classList.remove('hidden');
            return;
        }

        // Render Cards
        cardsGrid.innerHTML = '';
        data.recommendations.forEach(movie => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            card.innerHTML = `
                <div>
                    <div class="card-top">
                        <h3 class="movie-title">${escapeHtml(movie.title)}</h3>
                        <span class="score-chip">${movie.similarity_score}% Match</span>
                    </div>
                    <div class="movie-meta">
                        <span>${escapeHtml(movie.genre)}</span>
                        <span>•</span>
                        <span>${movie.release_year}</span>
                    </div>
                    <p class="movie-overview">${escapeHtml(movie.overview)}</p>
                </div>
                <div class="card-footer">
                    <span class="rating">★ ${movie.rating} / 10</span>
                    <span class="movie-id">ID: #${movie.id}</span>
                </div>
            `;
            cardsGrid.appendChild(card);
        });

        resultCount.textContent = `${data.recommendations.length} recommendations`;
        responseTime.textContent = `${duration}ms response`;
        resultsSection.classList.remove('hidden');

    } catch (err) {
        loadingState.classList.add('hidden');
        errorMessage.textContent = err.message || "Failed to communicate with recommendation server.";
        errorBanner.classList.remove('hidden');
    }
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
