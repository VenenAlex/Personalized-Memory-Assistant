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
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
GEMINI_MODEL = "gemini-2.0-flash"

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
def parse_txt_session(file_path: Path) -> Dict:
    """Parse old .txt session files into JSON format"""
    messages = []
    created_at = datetime.fromtimestamp(file_path.stat().st_ctime, tz=timezone.utc).isoformat()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse messages with regex to handle format: [timestamp] role: content
        pattern = r'\[([\d\-:T+\.]+)\]\s+(user|assistant):\s+(.+?)(?=\n\[|$)'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        for timestamp, role, content in matches:
            messages.append({
                "role": role.lower(),
                "content": content.strip(),
                "timestamp": timestamp
            })
        
        # Generate meaningful title from first user message
        title = None
        if messages:
            first_user_msg = next((m for m in messages if m['role'] == 'user'), None)
            if first_user_msg:
                # Extract first few words as title
                words = first_user_msg['content'].split()[:5]
                title = " ".join(words).capitalize()
                if len(first_user_msg['content']) > 50:
                    title = title[:47] + "..."
        
        if not title:
            title = "Chat Session"
            
    except Exception as e:
        print(f"Error parsing txt session {file_path}: {e}")
        title = "Chat Session"
    
    return {
        "session_id": file_path.stem,  # Use filename without extension as session_id
        "title": title,
        "created_at": created_at,
        "messages": messages
    }

def get_session_file_path(session_id: str) -> Path:
    """Get the file path for a session"""
    # Find existing session file with this ID (JSON first, then TXT)
    for file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("session_id") == session_id:
                    return file
        except:
            continue
    
    # Check if it's a txt file by exact filename match
    txt_file = SESSIONS_DIR / f"{session_id}.txt"
    if txt_file.exists():
        return txt_file
    
    # Create new session file
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SESSIONS_DIR / f"session_{session_id}_{ts}.json"

def load_session(session_id: str) -> Dict:
    """Load session data from file (supports both .json and .txt formats)"""
    file_path = get_session_file_path(session_id)
    
    if file_path.exists():
        try:
            # Try JSON first
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # Parse old TXT format
            elif file_path.suffix == '.txt':
                return parse_txt_session(file_path)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
    
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
    """Summarize YouTube video with transcript or Whisper fallback"""
    try:
        from yt_dlp import YoutubeDL
    except:
        return None, "yt_dlp not installed. Install with: pip install yt-dlp"

    try:
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'outtmpl': '%(id)s',
            'quiet': True,
            'no_warnings': True
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id', None)

        # Try to load transcript from downloaded file
        transcript_file = f"{video_id}.en.vtt"
        transcript_text = None
        
        if os.path.exists(transcript_file):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                transcript_text = ""
                for line in lines:
                    if '-->' not in line and not line.strip().isdigit() and line.strip() and "WEBVTT" not in line:
                        cleaned_line = re.sub(r'<[^>]+>', '', line).strip()
                        if cleaned_line:
                            transcript_text += " " + cleaned_line
                
                transcript_text = transcript_text.strip()
                
                # Clean up transcript file
                try:
                    os.remove(transcript_file)
                except:
                    pass
            except Exception as e:
                print(f"Error reading transcript file: {e}")
                transcript_text = None
        
        # If no transcript found, try Whisper fallback
        if not transcript_text:
            print(f"No transcript found for video {video_id}, trying Whisper fallback...")
            transcript_text, whisper_error = extract_audio_with_whisper(url)
            if not transcript_text:
                return None, f"No captions found and Whisper extraction failed: {whisper_error}"
        
        if not transcript_text:
            return None, "Could not extract transcript or audio from video"
        
        # Generate summary
        prompt = f"""Summarize this YouTube video transcript in a clear, concise way with key points:

{transcript_text[:4000]}

Provide a well-structured summary with main points and conclusions."""

        summary = generate_with_gemini(prompt, model="gemini-2.0-flash")
        
        if summary.startswith("Gemini error:") or summary.startswith("Gemini API key"):
            return None, summary
        
        return summary, None

    except Exception as e:
        error_msg = str(e)
        return None, f"Error processing YouTube video: {error_msg}"


def extract_audio_with_whisper(url: str) -> tuple:
    """Extract audio from YouTube video and transcribe using Whisper"""
    try:
        import whisper
        from yt_dlp import YoutubeDL
    except ImportError:
        return None, "whisper or yt_dlp not installed. Install with: pip install openai-whisper yt-dlp"
    
    temp_audio_path = None
    try:
        print(f"Downloading audio from YouTube: {url}")
        
        # Download audio from YouTube
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'temp_audio',
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        
        temp_audio_path = 'temp_audio.mp3'
        
        if not os.path.exists(temp_audio_path):
            return None, "Failed to download audio"
        
        print(f"Transcribing audio with Whisper...")
        
        # Load Whisper model (base model is good for speed/accuracy trade-off)
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(temp_audio_path, language="en", verbose=False)
        
        transcript = result.get('text', '').strip()
        
        if not transcript:
            return None, "Whisper could not transcribe audio"
        
        print(f"Transcription complete: {len(transcript)} characters")
        return transcript, None
    
    except Exception as e:
        error_msg = str(e)
        print(f"Whisper extraction error: {error_msg}")
        return None, error_msg
    
    finally:
        # Clean up temp files
        for temp_file in ['temp_audio.mp3', 'temp_audio.m4a']:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

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
    """Get all chat sessions (supports both .json and .txt formats)"""
    sessions = []
    all_files = []
    
    # Collect both JSON and TXT files
    all_files.extend(SESSIONS_DIR.glob("*.json"))
    all_files.extend(SESSIONS_DIR.glob("*.txt"))
    
    for file in sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            if file.suffix == '.json':
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append(SessionResponse(
                        session_id=data["session_id"],
                        title=data.get("title", "Untitled Chat"),
                        created_at=data["created_at"],
                        message_count=len(data.get("messages", []))
                    ))
            elif file.suffix == '.txt':
                # Parse TXT file
                data = parse_txt_session(file)
                # Use filename as session_id for txt files
                session_id = file.stem
                sessions.append(SessionResponse(
                    session_id=session_id,
                    title=data.get("title", "Chat Session"),
                    created_at=data["created_at"],
                    message_count=len(data.get("messages", []))
                ))
        except Exception as e:
            print(f"Error reading session file {file}: {e}")
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
    try:
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
        
        print(f"Generating response for: {request.message}")
        reply = generate_with_gemini(system_prompt)
        print(f"Generated reply: {reply[:100]}...")
        
        # Check if reply contains an error
        if reply.startswith("Gemini error:") or reply.startswith("Gemini API key not found"):
            print(f"ERROR: {reply}")
            raise HTTPException(status_code=500, detail=reply)
        
        # Save assistant message
        append_message(request.session_id, "assistant", reply)
        
        return {"reply": reply}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
    raise HTTPException(status_code=501, detail="Voice input not yet implemented. Use browser's Web Speech API instead.")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    """Handle file upload - supports video files"""
    try:
        # Debug: Print what we received
        print(f"DEBUG: Received file: {file.filename if file else 'None'}")
        print(f"DEBUG: Received session_id: {session_id}")
        
        # Save session reference
        if not session_id:
            print("ERROR: session_id is None/empty")
            raise HTTPException(status_code=400, detail="session_id is required")
        
        # Check file type
        filename = file.filename.lower()
        supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        
        is_video = any(filename.endswith(fmt) for fmt in supported_video_formats)
        
        if not is_video:
            raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported formats: {', '.join(supported_video_formats)}")
        
        # Save user message about upload
        user_message = f"Uploaded video file: {filename}"
        append_message(session_id, "user", user_message)
        
        # Process video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"DEBUG: Saved temp file to: {temp_file_path}")
        
        try:
            # Extract audio and transcribe
            print("DEBUG: Starting process_video_file...")
            transcript, error = process_video_file(temp_file_path)
            
            if error:
                error_msg = f"Error processing video: {error}"
                print(f"DEBUG: Video processing error: {error_msg}")
                append_message(session_id, "assistant", error_msg)
                return {"message": error_msg, "success": False}
            
            if not transcript or len(transcript.strip()) == 0:
                error_msg = "Transcription resulted in empty text"
                print(f"DEBUG: {error_msg}")
                append_message(session_id, "assistant", error_msg)
                return {"message": error_msg, "success": False}
            
            # Generate summary from transcript
            print(f"DEBUG: Generating summary from {len(transcript)} characters of transcript...")
            prompt = f"""Summarize this video transcript in a clear, concise way with key points:

{transcript[:4000]}

Provide a well-structured summary with main points."""
            
            summary = generate_with_gemini(prompt, model="gemini-2.0-flash")
            
            if not summary or "error" in summary.lower():
                print(f"DEBUG: Gemini summary failed: {summary}")
                append_message(session_id, "assistant", f"Error generating summary: {summary}")
                return {"message": f"Error generating summary: {summary}", "success": False}
            
            # Save summary
            response_message = f"Video Summary:\n\n{summary}"
            append_message(session_id, "assistant", response_message)
            
            print(f"DEBUG: Upload successful!")
            return {
                "message": response_message,
                "success": True,
                "filename": filename,
                "transcript_length": len(transcript)
            }
        
        finally:
            # Clean up temp files
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"DEBUG: Cleaned up temp file")
            except Exception as e:
                print(f"DEBUG: Error cleaning up: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"File upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def process_video_file(file_path: str) -> tuple:
    """Extract audio from video and transcribe it using FFmpeg + Whisper (no moviepy)"""
    temp_audio_path = None
    try:
        # Step 1: Extract audio from video using FFmpeg
        print(f"Processing video file: {file_path}")
        
        temp_audio_path = file_path.replace(Path(file_path).suffix, ".wav")
        
        # Build FFmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", file_path,
            "-q:a", "9",
            "-y",  # Overwrite output file
            temp_audio_path
        ]
        
        print(f"Extracting audio to: {temp_audio_path}")
        
        try:
            import subprocess
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "FFmpeg extraction failed"
                print(f"DEBUG: FFmpeg error: {error_msg}")
                return None, f"Audio extraction failed: {error_msg}"
        except FileNotFoundError:
            return None, "FFmpeg not found. Install FFmpeg from https://ffmpeg.org/download.html"
        except Exception as e:
            return None, f"FFmpeg error: {str(e)}"
        
        if not os.path.exists(temp_audio_path):
            return None, "Audio extraction failed - file not created"
        
        # Step 2: Transcribe audio with Whisper
        print("Transcribing audio with Whisper...")
        transcript, whisper_error = transcribe_with_whisper(temp_audio_path)
        
        if whisper_error:
            return None, f"Transcription failed: {whisper_error}"
        
        if not transcript:
            return None, "Could not transcribe audio"
        
        print(f"Transcription complete: {len(transcript)} characters")
        return transcript, None
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing video: {error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg
    
    finally:
        # Clean up temp audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary audio file")
            except:
                pass


def transcribe_with_whisper(audio_path: str) -> tuple:
    """Transcribe audio file using Whisper (local, no API limits)"""
    try:
        import whisper
    except ImportError:
        return None, "whisper not installed. Install with: pip install openai-whisper"
    
    try:
        if not os.path.exists(audio_path):
            return None, "Audio file not found"
        
        print(f"Loading Whisper model...")
        model = whisper.load_model("base", device="cpu")
        
        print(f"Transcribing audio (this may take a moment)...")
        result = model.transcribe(audio_path, language="en", verbose=False)
        
        transcript = result.get('text', '').strip()
        
        if not transcript:
            return None, "Whisper produced empty transcript"
        
        return transcript, None
    
    except Exception as e:
        error_msg = str(e)
        print(f"Whisper error: {error_msg}")
        return None, error_msg

if __name__ == "__main__":
    # Verify API key is loaded
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if (api_key):
        print(f"✓ API key loaded: {api_key[:20]}...")
    else:
        print("✗ WARNING: No GOOGLE_API_KEY or GEMINI_API_KEY found in environment!")
        print("  Check your .env file in the parent directory")
    
    print(f"✓ Using model: {GEMINI_MODEL}")
    print(f"✓ Sessions directory: {SESSIONS_DIR.absolute()}")
    print(f"✓ ChromaDB directory: {CHROMA_PERSIST_DIR.absolute()}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
