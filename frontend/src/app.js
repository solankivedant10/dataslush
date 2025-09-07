// frontend/src/App.js
import React, { useEffect, useState } from "react";
import "./app.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

function JobCard({ job, onSelect }) {
  return (
    <div className="job-card" onClick={() => onSelect(job)}>
      <h3>{job.title}</h3>
      <div className="meta">
        <small>{job.meta.location_pref ? `Location: ${job.meta.location_pref}` : ""}</small>
        <small>{job.meta.budget_monthly ? `Budget/month: $${job.meta.budget_monthly}` : ""}</small>
      </div>
      <div className="skills">{job.skills.join(", ")}</div>
    </div>
  );
}

// ⭐ CandidateRow now supports "Why this match?" using Gemini
function CandidateRow({ c, jobKey }) {
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchExplain = () => {
    setLoading(true);
    fetch(`${API_BASE}/explain/${jobKey}/${c.idx}`)
      .then(r => r.json())
      .then(json => {
        setExplanation(json.explanation);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  return (
    <div className="candidate-row">
      <div className="left">
        <strong>{c.name}</strong>
        <div className="sub">{c.city}, {c.country}</div>
      </div>
      <div className="right">
        <div className="score">Score: {c.final_score.toFixed(2)}</div>
        <details>
          <summary>Breakdown</summary>
          <div>Skill sim: {c.breakdown.skill_sim.toFixed(3)}</div>
          <div>Bio sim: {c.breakdown.bio_sim.toFixed(3)}</div>
          <div>Rate fit: {c.breakdown.rate_fit.toFixed(3)}</div>
          <div>Popularity: {c.breakdown.popularity.toFixed(3)}</div>
          <div>Pref: {c.breakdown.preference.toFixed(3)}</div>
        </details>

        {/* ⭐ New Explainability button */}
        <button onClick={fetchExplain} disabled={loading}>
          {loading ? "Explaining..." : "Why this match?"}
        </button>
        {explanation && <p className="explain-box">{explanation}</p>}
      </div>
    </div>
  );
}

function App() {
  const [jobs, setJobs] = useState([]);
  const [selected, setSelected] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/jobs`)
      .then(r => r.json())
      .then(setJobs)
      .catch(err => console.error(err));
  }, []);

  const fetchTop = (job) => {
    setSelected(job);
    setLoading(true);
    fetch(`${API_BASE}/recommend/${job.key}?k=10`)
      .then(r => r.json())
      .then(json => {
        setCandidates(json.results);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  };

  return (
    <div className="App">
      <header>
        <h1>Talent Recommender — DataSlush Demo</h1>
        <p>Click a job to see Top 10 candidates</p>
      </header>

      <main>
        <aside className="jobs-list">
          {jobs.map(j => (
            <JobCard key={j.key} job={j} onSelect={fetchTop} />
          ))}
        </aside>

        <section className="results">
          {selected ? (
            <div>
              <h2>{selected.title} — Top 10</h2>
              {loading ? (
                <div>Loading...</div>
              ) : (
                candidates.map((c) => (
                  <CandidateRow key={c.idx} c={c} jobKey={selected.key} /> // ⭐ Pass jobKey
                ))
              )}
            </div>
          ) : (
            <div className="placeholder">Select a job to get recommendations</div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
