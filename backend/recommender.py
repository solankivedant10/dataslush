# backend/recommender.py
"""
Recommender module
- Loads dataset
- Creates/loads embeddings
- Builds candidate vectors
- Exposes functions to get top-K recommendations for job posts
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Dict, Any

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "Talent Profiles - talent_samples.csv")
EMB_DIR = os.path.join(os.path.dirname(__file__), "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# Model name (small & fast)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Weights (tweakable)
W_SKILL = 0.6
W_BIO = 0.3
W_CREATORS = 0.1

# Scoring weights
WEIGHT_SKILL_SIM = 0.50
WEIGHT_BIO_SIM = 0.25
WEIGHT_RATE_FIT = 0.15
WEIGHT_POPULARITY = 0.05
WEIGHT_PREF_BOOST = 0.05

# Basic job posts (same as notebook)
JOB_POSTS = {
    "job_1": {
        "id": "UCi2qHfRMVEI_yHH90gZBevQ",
        "title": "Video Editor",
        "skills": ["adobe premiere pro", "splice & dice", "rough cut & sequencing", "2d animation"],
        "bio_keywords": ["entertainment", "lifestyle", "vlogs"],
        "location_pref": "asia",
        "budget_monthly": 2500
    },
    "job_2": {
        "id": "imjennim",
        "title": "Producer/Video Editor",
        "skills": ["storyboarding", "sound designing", "rough cut & sequencing", "filming", "tiktok"],
        "bio_keywords": ["entertainment", "education", "food & cooking"],
        "location_pref": ["new york","us_remote"],
        "budget_hourly": [100,150],
        "pref_gender": "female"
    },
    "job_3": {
        "id": "aliabdaal",
        "title": "Chief Operating Officer",
        "skills": ["strategy", "business operations", "development"],
        "bio_keywords": ["productivity","education","high energy","passion for education"],
        "location_pref": "any",
        "budget_unlimited": True
    }
}

# Utilities
def clean_text(x: Any) -> str:
    if pd.isnull(x): return ""
    return str(x).lower().strip()

def list_to_text(x):
    if pd.isnull(x): return ""
    if isinstance(x, list):
        return ", ".join(x)
    return str(x)

class Recommender:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.cand_vectors = None
        self.skill_emb = None
        self.bio_emb = None
        self.creators_emb = None
        self.max_views = 1.0
        self._load_data_and_embeddings()

    def _load_data_and_embeddings(self):
        # load dataframe
        df = pd.read_csv(DATA_CSV)
        # basic preprocessing for fields used
        df['Profile Description'] = df['Profile Description'].fillna('')
        df['Skills'] = df['Skills'].fillna('')
        df['Software'] = df['Software'].fillna('')
        df['Content Verticals'] = df['Content Verticals'].fillna('')
        df['Past Creators'] = df['Past Creators'].fillna('')
        df['Monthly Rate'] = pd.to_numeric(df['Monthly Rate'], errors='coerce').fillna(df['Monthly Rate'].median())
        df['Hourly Rate'] = pd.to_numeric(df['Hourly Rate'], errors='coerce').fillna(df['Hourly Rate'].median())

        # some derived fields
        df['bio_text'] = df['Profile Description'].astype(str)
        df['skill_text'] = df['Skills'].astype(str) + " | " + df['Software'].astype(str) + " | " + df['Content Verticals'].astype(str)
        df['creators_text'] = df['Past Creators'].astype(str)

        self.df = df
        self.max_views = df['# of Views by Creators'].max() if '# of Views by Creators' in df.columns else 1.0

        # try load cached embeddings
        skill_path = os.path.join(EMB_DIR, "skill_emb.npy")
        bio_path = os.path.join(EMB_DIR, "bio_emb.npy")
        creators_path = os.path.join(EMB_DIR, "creators_emb.npy")
        if os.path.exists(skill_path) and os.path.exists(bio_path) and os.path.exists(creators_path):
            self.skill_emb = np.load(skill_path)
            self.bio_emb = np.load(bio_path)
            self.creators_emb = np.load(creators_path)
        else:
            # compute embeddings and save
            print("Generating embeddings — this may take a minute...")
            self.skill_emb = self.model.encode(self.df['skill_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
            self.bio_emb = self.model.encode(self.df['bio_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
            self.creators_emb = self.model.encode(self.df['creators_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
            np.save(skill_path, self.skill_emb)
            np.save(bio_path, self.bio_emb)
            np.save(creators_path, self.creators_emb)

        # normalize
        self.skill_emb = normalize(self.skill_emb)
        self.bio_emb = normalize(self.bio_emb)
        self.creators_emb = normalize(self.creators_emb)

        # combine candidate vectors
        self.cand_vectors = np.vstack([
            self._combine(self.skill_emb[i], self.bio_emb[i], self.creators_emb[i])
            for i in range(len(self.df))
        ])

    def _combine(self, s, b, c):
        combined = W_SKILL * s + W_BIO * b + W_CREATORS * c
        if np.linalg.norm(combined) == 0:
            return combined
        return combined / np.linalg.norm(combined)

    # scoring helpers
    def _cos_sim(self, a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _rate_score(self, row, job):
        # returns 0..1
        if job.get('budget_monthly') is not None:
            pref = job['budget_monthly']
            cand = row.get('Monthly Rate', np.nan)
            if np.isnan(cand): return 0.5
            if cand <= pref: return 1.0
            return max(0.0, min(1.0, pref / cand))
        if job.get('budget_hourly') is not None:
            low, high = job['budget_hourly']
            cand = row.get('Hourly Rate', np.nan)
            if np.isnan(cand): return 0.5
            if low <= cand <= high: return 1.0
            diff = min(abs(cand - low), abs(cand - high))
            denom = max(low, high)
            return max(0.0, 1 - (diff / denom))
        if job.get('budget_unlimited'):
            return 1.0
        return 0.5

    def _passes_hard_filters(self, row, job):
        # job_2 is stricter (NY/US); others are soft preferences
        job_loc = job.get('location_pref', 'any')
        city = clean_text(row.get('City', ''))
        country = clean_text(row.get('Country', ''))
        if isinstance(job_loc, list):
            # require US-based or new york city
            if 'new york' in city or 'new york' in country:
                return True
            if 'us' in country or 'united states' in country or 'united states of america' in country:
                return True
            return False
        return True

    def _preference_boost(self, row, job):
        boost = 0.0
        if job.get('pref_gender'):
            if clean_text(row.get('Gender','')) == clean_text(job['pref_gender']):
                boost += 1.0
        if job.get('location_pref') == 'asia':
            asia_list = ["india","indonesia","philippines","pakistan","bangladesh","sri lanka","thailand","vietnam","malaysia","china","japan","south korea","nepal"]
            if any(a in clean_text(row.get('Country','')) for a in asia_list):
                boost += 0.5
        return min(1.0, boost/1.5)

    def get_top_k_for_job(self, job_key: str, k: int = 10) -> List[Dict[str, Any]]:
        if job_key not in JOB_POSTS:
            raise ValueError("Unknown job key")
        job = JOB_POSTS[job_key]
        # prepare job skill & bio embeddings (local)
        job_skill_text = job.get('title','') + " | " + ", ".join(job.get('skills', []))
        job_bio_text = " ".join(job.get('bio_keywords', []))
        job_skill_v = self.model.encode(job_skill_text, convert_to_numpy=True)
        job_bio_v = self.model.encode(job_bio_text, convert_to_numpy=True)
        if np.linalg.norm(job_skill_v) != 0:
            job_skill_v = job_skill_v / np.linalg.norm(job_skill_v)
        if np.linalg.norm(job_bio_v) != 0:
            job_bio_v = job_bio_v / np.linalg.norm(job_bio_v)

        results = []
        for i, row in self.df.iterrows():
            if not self._passes_hard_filters(row, job):
                continue
            skill_sim = self._cos_sim(self.skill_emb[i], job_skill_v)
            bio_sim = self._cos_sim(self.bio_emb[i], job_bio_v)
            rscore = self._rate_score(row, job)
            views = row.get('# of Views by Creators', 0)
            pop_score = float(min(1.0, views / float(self.max_views)))
            pref_score = self._preference_boost(row, job)

            final_raw = (WEIGHT_SKILL_SIM * skill_sim +
                         WEIGHT_BIO_SIM * bio_sim +
                         WEIGHT_RATE_FIT * rscore +
                         WEIGHT_POPULARITY * pop_score +
                         WEIGHT_PREF_BOOST * pref_score)
            final_score = float(final_raw * 100)

            result = {
                "idx": int(i),
                "name": f"{row.get('First Name','')} {row.get('Last Name','')}",
                "city": row.get('City',''),
                "country": row.get('Country',''),
                "monthly_rate": float(row.get('Monthly Rate', np.nan)),
                "hourly_rate": float(row.get('Hourly Rate', np.nan)),
                "final_score": final_score,
                "breakdown": {
                    "skill_sim": skill_sim,
                    "bio_sim": bio_sim,
                    "rate_fit": rscore,
                    "popularity": pop_score,
                    "preference": pref_score
                }
            }
            results.append(result)

        # sort and return top-k
        results_sorted = sorted(results, key=lambda x: x['final_score'], reverse=True)
        return results_sorted[:k]

# Create a single instance (singleton) to reuse model & embeddings
RECOMMENDER = Recommender()
