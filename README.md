# Talent Recommender — DataSlush Demo

A prototype AI-driven talent recommendation engine that matches candidates to job postings using semantic search and LLM-powered explanations. Built with FastAPI (Python) and React (JavaScript).

---

## Features

- Upload and view job postings with requirements
- Candidate ranking based on skills, bio, rates, and semantic embeddings
- "Why this match?" explanations powered by Gemini API
- Interactive React UI with score breakdowns
- Local embeddings for fast semantic search

---

## Tech Stack

- **Backend:** FastAPI, Sentence Transformers, Gemini API
- **Frontend:** React (Create React App)
- **Data:** CSV, Numpy embeddings
- **Other:** Python 3.10+, Node.js 18+, Chrome browser

---

## Note on Gemini API
This project uses Google Gemini API for AI responses.
To run it fully, please create your own API key:
1. Visit https://aistudio.google.com/app/apikey
2. Copy the key and create a `.env` file in the backend folder:
   GEMINI_API_KEY=your_key_here
Without this key, the backend will still run, but AI features will not work.

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+ and npm](https://nodejs.org/)
- [Chrome browser](https://www.google.com/chrome/) (for demo)
- (Optional) [Git](https://git-scm.com/)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dataslush-talent-recommender.git
cd dataslush-talent-recommender
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Create a .env file in backend/ with your Gemini API key:
echo GEMINI_API_KEY=your_gemini_api_key > .env

# Run the FastAPI server
uvicorn main:app --reload
```

By default, the backend runs at [http://localhost:8000](http://localhost:8000).

### 3. Frontend Setup

Open a new terminal window:

```bash
cd frontend
npm install

# Create a .env file in frontend/ with the backend API URL:
echo REACT_APP_API_URL=http://localhost:8000 > .env

npm start
```

The frontend will open at [http://localhost:3000](http://localhost:3000).

---

## Demo

- [Add screenshots or a link to your demo video here]

---

## Data Schema & Cleaning

- **Input:**  
  - Job postings: CSV with fields for title, requirements, etc.
  - Talent profiles: CSV with fields for name, skills, bio, rate, etc.
- **Cleaning:**  
  - Text normalization, removal of stopwords, and embedding generation using Sentence Transformers.
  - Embeddings are precomputed and stored as `.npy` files for fast lookup.

---

## Scoring Formula & Weights

- **Similarity:**  
  - Cosine similarity between job and candidate embeddings (skills, bio).
- **Other factors:**  
  - Rate matching, skill overlap, and custom weights (see `recommender.py`).
- **Explanation:**  
  - Gemini API generates a natural language explanation for each match.

---

## Models Used

- **Sentence Transformers:**  
  - For semantic embedding of job and candidate text.
- **Gemini API:**  
  - For generating "Why this match?" explanations.

---

## Limitations & Next Steps

- **Limitations:**  
  - No real-time feedback loop or AB testing.
  - No persistent database (all data is local).
  - Gemini API key required for explanations.
- **Next Steps:**  
  - Add user feedback and AB testing.
  - Integrate persistent storage (e.g., MongoDB).
  - Deploy online for broader access.

---

**Notes:**
- Ensure `.env` files are **not** committed to git.
- If you encounter issues, check that both backend and frontend servers are running and that API URLs match your local setup.
