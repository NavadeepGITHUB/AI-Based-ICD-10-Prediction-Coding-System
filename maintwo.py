from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import re


from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

DATABASE_URL = "sqlite:///./medical_coding.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id = Column(Integer, primary_key=True, index=True)
    note_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate_codes = relationship("CandidateCode", back_populates="note")


class CandidateCode(Base):
    __tablename__ = "candidate_codes"

    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(Integer, ForeignKey("clinical_notes.id"), nullable=False)
    code = Column(String(10), nullable=False)
    reason = Column(Text, nullable=False)

    note = relationship("ClinicalNote", back_populates="candidate_codes")

Base.metadata.create_all(bind=engine)

ICD_KEYWORDS = {
    "asthma": "J45.9",
    "hypertension": "I10",
    "diabetes": "E11",
    "stroke": "I63.9",
    "lung cancer": "C34.9",
    "mild cough": "R05.1",
    "severe cough": "R05.9",
    "cough unspecified": "R05.9",  
}

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HISTORICAL_NOTES = [
    "Patient presents with asthma history and wheezing.",
    "Long standing hypertension, high blood pressure noted.",
    "Type 2 diabetes with high fasting sugar levels.",
    "Severe headache due to brain stroke in past.",
    "Suspicious mass in lung, possible lung cancer.",
    "Patient reports mild cough for the past three days.",
    "Severe cough with persistent chest irritation observed.",
    "Dry cough with throat discomfort and mild fever.",
    "Chronic cough worsening at night time.",
]

vectorizer = TfidfVectorizer()

def preprocess(text: str) -> str:
    """Lowercase, remove symbols, clean spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

processed_hist = [preprocess(t) for t in HISTORICAL_NOTES]
historical_matrix = vectorizer.fit_transform(processed_hist)

def retrieve_similar_notes(note: str):
    """Return the most similar historical note and similarity score using TF-IDF."""
    processed = preprocess(note)
    query_vec = vectorizer.transform([processed])
    sims = cosine_similarity(query_vec, historical_matrix)[0]
    best_idx = int(sims.argmax())
    return HISTORICAL_NOTES[best_idx], float(sims[best_idx])

def generate_codes(note: str):
    """Simple 'generation' using dictionary logic (ICD-10 suggestions)."""
    processed = preprocess(note)
    codes = []
    for keyword, icd_code in ICD_KEYWORDS.items():
        if keyword in processed:
            codes.append(
                {
                    "code": icd_code,
                    "keyword": keyword, 
                    "reason": f'keyword "{keyword}" matched in note',
                }
            )
    return codes

def find_evidence_for_keyword(keyword: str):
    """
    For a given disease keyword, return a matching historical note
    and a similarity score. First try exact keyword match, otherwise
    fall back to TF-IDF.
    """
    for hist in HISTORICAL_NOTES:
        if keyword in preprocess(hist):
            return hist, 1.0 
    return retrieve_similar_notes(keyword)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI-Based ICD-10 Prediction and Coding System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    note: str
class CandidateCodeOut(BaseModel):
    code: str
    reason: str
    evidence: str
    similarity: float
class PredictResponse(BaseModel):
    note_id: int
    candidate_codes: List[CandidateCodeOut]

@app.get("/debug/keywords")
def debug_keywords():
    """Helper endpoint to see which ICD keywords the server is using."""
    return ICD_KEYWORDS

@app.post("/predict", response_model=PredictResponse)
def predict_codes(payload: PredictRequest):
    """
    MAIN PIPELINE (RAG-style):

    1. Store doctor's note.
    2. Run NLP preprocessing.
    3. Generate ICD-10 codes using dictionary rules.
    4. For each code, retrieve an evidence note (RAG retrieval).
    5. Store candidate codes.
    6. Return JSON with per-code evidence and similarity.
    """
    db = SessionLocal()
    try:

        note_obj = ClinicalNote(note_text=payload.note)
        db.add(note_obj)
        db.commit()
        db.refresh(note_obj)

        codes = generate_codes(payload.note)
        if not codes:
            raise HTTPException(
                status_code=400,
                detail="No supported disease keywords found in this note.",
            )

        response_codes: List[CandidateCodeOut] = []
        for c in codes:
            cc = CandidateCode(
                note_id=note_obj.id,
                code=c["code"],
                reason=c["reason"],
            )
            db.add(cc)

            evidence_note, ev_sim = find_evidence_for_keyword(c["keyword"])
            response_codes.append(
                CandidateCodeOut(
                    code=c["code"],
                    reason=c["reason"],
                    evidence=evidence_note,
                    similarity=ev_sim,
                )
            )
        db.commit()
        return PredictResponse(
            note_id=note_obj.id,
            candidate_codes=response_codes,
        )
    finally:
        db.close()
