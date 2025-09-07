# Talent Recommender — DataSlush Demo

🚀 A prototype AI-driven talent recommendation engine with FastAPI backend + React frontend.

## Demo
- Backend: [Render](https://dataslush-backend.onrender.com/jobs)
- Frontend: [Vercel](https://dataslush-frontend.vercel.app)

## Features
- Job postings with requirements
- Candidate ranking (skills, bio, rates, embeddings)
- "Why this match?" explanation powered by Gemini
- Interactive React UI with score breakdown

## Tech
- Backend: FastAPI, Sentence Transformers, Gemini API
- Frontend: React (Vercel)
- Deployment: Render (backend), Vercel (frontend)

## Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm start
