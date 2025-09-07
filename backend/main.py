import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataslush")
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from recommender import RECOMMENDER, JOB_POSTS
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai   # Gemini SDK

# Load .env file
load_dotenv()

# Configure Gemini
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

app = FastAPI(title="Talent Recommender API")

# Allow CORS from frontend (change origin in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- MODELS --------
class JobOut(BaseModel):
    key: str
    id: str
    title: str
    skills: List[str]
    bio_keywords: List[str]
    meta: Dict[str, Any] = {}

# -------- ENDPOINTS --------
@app.get("/jobs", response_model=List[JobOut])
def get_jobs():
    """Return available job postings."""
    out = []
    for k, v in JOB_POSTS.items():
        out.append({
            "key": k,
            "id": v.get("id"),
            "title": v.get("title"),
            "skills": v.get("skills", []),
            "bio_keywords": v.get("bio_keywords", []),
            "meta": {
                "location_pref": v.get("location_pref"),
                "budget_monthly": v.get("budget_monthly"),
                "budget_hourly": v.get("budget_hourly", None)
            }
        })
    return out


@app.get("/recommend/{job_key}")
def recommend(job_key: str, k: int = 10):
    try:
        res = RECOMMENDER.get_top_k_for_job(job_key, k=k)
        return {"job_key": job_key, "results": res}
    except ValueError:
        raise HTTPException(status_code=404, detail="job_key not found")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/explain/{job_key}/{cand_idx}")
def explain(job_key: str, cand_idx: int):
    """Generate a natural-language explanation for candidate-job fit using Gemini."""
    if not GEMINI_KEY:
        raise HTTPException(500, "Gemini API key not set in environment variable GEMINI_API_KEY")

    if job_key not in JOB_POSTS:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        job = JOB_POSTS[job_key]
        row = RECOMMENDER.df.iloc[cand_idx]
        # also re-run scoring for context
        top_candidates = RECOMMENDER.get_top_k_for_job(job_key, 50)
        candidate_data = next((c for c in top_candidates if c["idx"] == cand_idx), None)

        if not candidate_data:
            raise HTTPException(404, detail="Candidate not found in top results")

        # Build prompt
        prompt = f"""
        You are an assistant explaining candidate-job fit.

        Job:
        - Title: {job['title']}
        - Required skills: {', '.join(job.get('skills', []))}
        - Keywords: {', '.join(job.get('bio_keywords', []))}

        Candidate:
        - Name: {row['First Name']} {row['Last Name']}
        - Location: {row['City']}, {row['Country']}
        - Skills: {row['Skills']}
        - Bio: {row['Profile Description']}
        - Rates: Monthly {row['Monthly Rate']}, Hourly {row['Hourly Rate']}

        Matching breakdown (numerical):
        {candidate_data['breakdown']}

        Write a short, friendly explanation (3–5 sentences) summarizing why this candidate is or isn’t a good fit for the role.
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return {
            "candidate": f"{row['First Name']} {row['Last Name']}",
            "job": job["title"],
            "explanation": response.text
        }

    except Exception as e:
        raise HTTPException(500, str(e))
