#!/usr/bin/env python3
"""
FastAPI Backend for Personalized Memory Assistant
Modern ChatGPT-like interface with intelligent session naming
"""
import os
import re
import uuid
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
SESSIONS_DIR = Path("../sessions")
CHROMA_PERSIST_DIR = Path("../chroma_db")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Load API key from .env file in parent directory
dotenv_path = Path("../.env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)

SESSIONS_DIR.mkdir(exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str

class SummarizeRequest(BaseModel):
    url: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    message_count: int

# ChromaDB Memory Store
class ChromaMemoryStore:
    def __init__(self, collection_name="chat_memories"):
        try:
            self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        except Exception:
            settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(CHROMA_PERSIST_DIR),
            )
            self.client = chromadb.Client(settings=settings)

        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.lock = threading.Lock()

    def add_text(self, text, source_file=None, metadata=None):
        if metadata is None:
            metadata = {}
        if source_file:
            metadata["source"] = str(source_file)

        emb = self.embed_model.encode(text, convert_to_numpy=True).tolist()
        uid = str(uuid.uuid4())
        with self.lock:
            self.collection.add(
                documents=[text],
                embeddings=[emb],
                ids=[uid],
                metadatas=[metadata],
            )

    def query(self, text, top_k=TOP_K):
        q_emb = self.embed_model.encode(text, convert_to_numpy=True).tolist()
        with self.lock:
            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
            )
        hits = []
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for doc, dist, md in zip(docs, distances, metadatas):
            hits.append({"text": doc, "score": float(dist), "metadata": md})
        return hits

# Initialize memory store
memory_store = ChromaMemoryStore()

# Gemini API integration
def generate_with_gemini(prompt, model=GEMINI_MODEL, temperature=0.7):
    try:
        import google.generativeai as genai
    except Exception:
        return "Gemini client not installed."

    # Try multiple API key environment variables
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("VERTEX_API_KEY")
    if not api_key:
        return "Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in .env file."

    try:
        genai.configure(api_key=api_key)
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        return f"Gemini error: {e}"

# Generate intelligent session title
def generate_session_title(first_message: str) -> str:
    """Generate a ChatGPT-like title from the first message using Gemini"""
    prompt = f"""Generate a short, descriptive title (3-5 words max) for a chat conversation that starts with this message:
"{first_message}"

Rules:
- Maximum 5 words
- Capitalize first letter of each word
- No quotes or special characters
- Be specific and descriptive
- Examples: "Python Web Scraping Help", "Recipe For Chocolate Cake", "Math Homework Assistance"

Title:"""
    
    title = generate_with_gemini(prompt, temperature=0.5)
    # Clean up the title
    title = title.strip().replace('"', '').replace("'", "")[:50]
    if not title or "error" in title.lower():
        # Fallback: use first few words of message
        words = first_message.split()[:4]
        title = " ".join(words).capitalize()
    return title

# Session management
def get_session_file_path(session_id: str) -> Path:
    """Get the file path for a session"""
    # Find existing session file with this ID
    for file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("session_id") == session_id:
                    return file
        except:
            continue
    
    # Create new session file
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SESSIONS_DIR / f"session_{session_id}_{ts}.json"

def load_session(session_id: str) -> Dict:
    """Load session data from file"""
    file_path = get_session_file_path(session_id)
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    # Create new session
    return {
        "session_id": session_id,
        "title": "New Chat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": []
    }

def save_session(session_data: Dict):
    """Save session data to file"""
    session_id = session_data["session_id"]
    file_path = get_session_file_path(session_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

def append_message(session_id: str, role: str, content: str):
    """Append a message to the session"""
    session_data = load_session(session_id)
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    session_data["messages"].append(message)
    
    # Generate title from first user message
    if role == "user" and len(session_data["messages"]) == 1:
        session_data["title"] = generate_session_title(content)
    
    save_session(session_data)
    
    # Add to ChromaDB
    try:
        memory_store.add_text(
            f"{role}: {content}",
            source_file=get_session_file_path(session_id),
            metadata={"role": role, "session_id": session_id}
        )
    except Exception as e:
        print(f"Error adding to ChromaDB: {e}")

# YouTube summarizer
def summarize_youtube(url: str) -> tuple:
    """Summarize YouTube video"""
    try:
        from yt_dlp import YoutubeDL
    except:
        return None, "yt_dlp not installed"

    try:
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'outtmpl': '%(id)s',
            'quiet': True
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id', None)

        transcript_file = f"{video_id}.en.vtt"
        if not os.path.exists(transcript_file):
            return None, "No English transcript found"

        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        transcript_text = ""
        for line in lines:
            if '-->' not in line and not line.strip().isdigit() and line.strip() and "WEBVTT" not in line:
                cleaned_line = re.sub(r'<[^>]+>', '', line).strip()
                transcript_text += " " + cleaned_line

        try:
            os.remove(transcript_file)
        except:
            pass

        prompt = f"""Summarize this YouTube video transcript in a clear, concise way with key points:

{transcript_text[:4000]}

Provide a well-structured summary with main points."""

        summary = generate_with_gemini(prompt, model="gemini-2.0-flash-exp")
        return summary, None

    except Exception as e:
        return None, str(e)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Personalized Memory Assistant API"}

@app.post("/api/session/new")
async def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "title": "New Chat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": []
    }
    save_session(session_data)
    return {"session_id": session_id}

@app.get("/api/sessions")
async def get_sessions() -> List[SessionResponse]:
    """Get all chat sessions"""
    sessions = []
    for file in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sessions.append(SessionResponse(
                    session_id=data["session_id"],
                    title=data.get("title", "Untitled Chat"),
                    created_at=data["created_at"],
                    message_count=len(data.get("messages", []))
                ))
        except:
            continue
    return sessions

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get a specific session's messages"""
    session_data = load_session(session_id)
    return session_data

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    file_path = get_session_file_path(session_id)
    if file_path.exists():
        file_path.unlink()
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat message"""
    # Save user message
    append_message(request.session_id, "user", request.message)
    
    # Query similar memories
    hits = memory_store.query(request.message, top_k=TOP_K)
    context_lines = []
    for h in hits:
        context_lines.append(f"- {h['text']}")
    context = "\n".join(context_lines) if context_lines else "No relevant memories."
    
    # Generate response
    system_prompt = f"""You are a helpful AI assistant with access to conversation history and memories.

Relevant memories:
{context}

User message: {request.message}

Provide a helpful, concise response."""
    
    reply = generate_with_gemini(system_prompt)
    
    # Save assistant message
    append_message(request.session_id, "assistant", reply)
    
    return {"reply": reply}

@app.post("/api/summarize")
async def summarize(request: SummarizeRequest):
    """Summarize YouTube video"""
    # Save user request
    append_message(request.session_id, "user", f"Summarize: {request.url}")
    
    summary, error = summarize_youtube(request.url)
    
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    # Save summary
    append_message(request.session_id, "assistant", summary)
    
    return {"summary": summary}

@app.post("/api/voice")
async def voice_input(file: UploadFile = File(...)):
    """Handle voice input (placeholder - requires speech recognition setup)"""
    # This is a placeholder - you'd need to implement actual speech recognition
    # For now, we'll return an error
    raise HTTPException(status_code=501, detail="Voice input not yet implemented. Use browser's Web Speech API instead.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
