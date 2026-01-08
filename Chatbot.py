#!/usr/bin/env python3
"""
chat.py
Voice+Text chatbot with per-session .txt files and persistent ChromaDB memory.
Saves every user message to a session file and adds only new messages to ChromaDB.
Generates replies with Google Gemini (google-generativeai). Works with both
old (text.generate) and newer (chat.completions.create) client APIs.

ENHANCED WITH: YouTube transcript extraction using YouTube API + Whisper fallback
"""
import re
import os
import time
import uuid
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --------- Configuration ---------
SESSIONS_DIR = Path("sessions")
CHROMA_PERSIST_DIR = Path("chroma_db")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
GEMINI_MODEL = "gemini-2.0-flash"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
AUDIO_FORMAT = "mp3"

import chromadb
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
import pyttsx3

try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None

try:
    import subprocess
    FFMPEG_AVAILABLE = True
except Exception:
    FFMPEG_AVAILABLE = False

# Import YouTube transcript extraction libraries
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except Exception:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# ----------------- Chroma store -----------------
class ChromaMemoryStore:
    def __init__(self, collection_name="chat_memories"):
        # Use PersistentClient if available; fallback to Client with settings.
        # Many chromadb versions expose PersistentClient; adapt if not.
        try:
            # prefer PersistentClient API
            self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        except Exception:
            # fallback
            settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(CHROMA_PERSIST_DIR),
            )
            self.client = chromadb.Client(settings=settings)

        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.lock = threading.Lock()

    def add_text(self, text, source_file=None, metadata=None):
        """
        Add a single text item to Chroma with its embedding.
        source_file: path of session file (string) to store as metadata
        metadata: optional dict (merged into stored metadata)
        """
        if metadata is None:
            metadata = {}
        if source_file:
            metadata["source"] = str(source_file)

        emb = self.embed_model.encode(text, convert_to_numpy=True).tolist()
        uid = str(uuid.uuid4())
        with self.lock:
            # chromadb.collection.add accepts documents, embeddings, ids, metadatas
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


def generate_with_gemini(prompt, model=GEMINI_MODEL, temperature=0.7, max_tokens=512):
    try:
        import google.generativeai as genai
    except Exception:
        return "Gemini client not installed. pip install google-generativeai"

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("VERTEX_API_KEY")
    if not api_key:
        return "Gemini API key not found. Set GOOGLE_API_KEY or VERTEX_API_KEY env var."

    try:
        genai.configure(api_key=api_key)
    except Exception:
        try:
            genai.api_key = api_key
        except Exception:
            pass

    try:
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        return str(response)
    except Exception as e:
        return f"Gemini call failed: {e}"


# Voice & TTS helpers
def listen_from_mic(timeout=5, phrase_time_limit=10):
    if sr is None:
        raise RuntimeError("SpeechRecognition not installed. pip install SpeechRecognition and pyaudio.")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... speak now.")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        raise RuntimeError(f"Speech recognition failed: {e}")

def speak_text(text):
    if pyttsx3 is None:
        print("(TTS not available) " + text)
        return
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except KeyboardInterrupt:
        print("\n(TTS interrupted by user)")
    except Exception as e:
        print("(TTS error)", e)

# Session file helpers
def new_session_file(prefix="session"):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = SESSIONS_DIR / f"{prefix}_{ts}.txt"
    # create an empty file
    fname.touch(exist_ok=False)
    return fname

def append_to_session(file_path: Path, role: str, text: str):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {role}: {text}\n"
    file_path.open("a", encoding="utf-8").write(line)

# ============================================================================
# ENHANCED YOUTUBE TRANSCRIPT EXTRACTION
# ============================================================================

def get_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    try:
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url

        return None

    except Exception as e:
        print(f"❌ Error extracting video ID: {str(e)}")
        return None


def get_youtube_transcript(video_id: str) -> Optional[str]:
    """Fetch existing transcript/captions from YouTube."""
    if not YOUTUBE_TRANSCRIPT_AVAILABLE:
        print("ℹ️  youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")
        return None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])

        print(f"✅ YouTube transcript found for video: {video_id}")
        return transcript_text

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"ℹ️  No YouTube transcript available: {str(e)}")
        return None

    except Exception as e:
        print(f"❌ Error fetching transcript: {str(e)}")
        return None


def download_audio_from_youtube(url: str, temp_dir: str = None) -> Optional[str]:
    """Download audio from YouTube video using yt-dlp."""
    if YoutubeDL is None:
        print("❌ yt-dlp not installed. Install with: pip install yt-dlp")
        return None

    try:
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()

        video_id = get_video_id(url)
        if not video_id:
            print("❌ Invalid YouTube URL")
            return None

        output_path = os.path.join(temp_dir, f"yt_audio_{video_id}.{AUDIO_FORMAT}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': AUDIO_FORMAT,
                'preferredquality': '192',
            }],
            'outtmpl': output_path.replace(f'.{AUDIO_FORMAT}', ''),
            'quiet': True,
            'no_warnings': True,
        }

        print(f"⬇️  Downloading audio from YouTube...")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists(output_path):
            print(f"✅ Audio downloaded: {output_path}")
            return output_path
        else:
            print(f"❌ Audio file not found after download")
            return None

    except Exception as e:
        print(f"❌ Error downloading audio: {str(e)}")
        return None


def generate_transcript_from_audio(audio_path: str) -> Optional[str]:
    """Generate transcript from audio file using Whisper (local)."""
    if not WHISPER_AVAILABLE:
        print("❌ Whisper not installed. Install with: pip install openai-whisper torch")
        return None

    try:
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device.upper()}")

        # Load Whisper model
        print(f"⚙️  Loading Whisper model: {WHISPER_MODEL}")
        model = whisper.load_model(WHISPER_MODEL, device=device)

        # Transcribe audio
        print(f"🎤 Transcribing audio... (this may take a few minutes)")
        result = model.transcribe(audio_path, fp16=(device == "cuda"))

        transcript_text = result["text"].strip()
        print(f"✅ Transcription complete ({len(transcript_text)} characters)")

        return transcript_text

    except Exception as e:
        print(f"❌ Error during transcription: {str(e)}")
        return None


def cleanup_audio_file(audio_path: str) -> None:
    """Remove temporary audio file after processing."""
    try:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"🗑️  Cleaned up: {audio_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not delete {audio_path}: {str(e)}")


def fetch_final_transcript(youtube_url: str) -> Dict[str, Optional[str]]:
    """
    Master function to fetch transcript from YouTube (with Whisper fallback).

    Complete workflow:
    1. Extract video ID
    2. Try YouTube transcript API
    3. If unavailable, download audio with yt-dlp
    4. Transcribe with Whisper (GPU accelerated if available)
    5. Clean up temporary files
    6. Return transcript with metadata

    Args:
        youtube_url: YouTube video URL

    Returns:
        Dictionary containing:
        - transcript: Full transcript text
        - method_used: "youtube" or "whisper"
        - video_id: YouTube video ID
        - error: Error message if failed (None if successful)
    """
    result = {
        "transcript": None,
        "method_used": None,
        "video_id": None,
        "error": None
    }

    audio_path = None

    try:
        # Step 1: Extract video ID
        print(f"\n{'='*70}")
        print(f"🎬 Processing: {youtube_url}")
        print(f"{'='*70}\n")

        video_id = get_video_id(youtube_url)
        if not video_id:
            result["error"] = "Invalid YouTube URL"
            return result

        result["video_id"] = video_id

        # Step 2: Try YouTube transcript API first
        print("📝 Attempting to fetch YouTube transcript...")
        transcript = get_youtube_transcript(video_id)

        if transcript:
            result["transcript"] = transcript
            result["method_used"] = "youtube"
            print(f"\n✅ SUCCESS: Transcript fetched via YouTube API\n")
            return result

        # Step 3: Fallback to Whisper transcription
        print("\n🎵 YouTube transcript not available. Using Whisper fallback...\n")

        # Download audio
        audio_path = download_audio_from_youtube(youtube_url)
        if not audio_path:
            result["error"] = "Failed to download audio"
            return result

        # Transcribe with Whisper
        transcript = generate_transcript_from_audio(audio_path)
        if not transcript:
            result["error"] = "Failed to transcribe audio"
            return result

        result["transcript"] = transcript
        result["method_used"] = "whisper"
        print(f"\n✅ SUCCESS: Transcript generated via Whisper\n")

        return result

    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        print(f"\n❌ FAILED: {result['error']}\n")
        return result

    finally:
        # Always clean up temporary audio file
        if audio_path:
            cleanup_audio_file(audio_path)


# --- ENHANCED YouTube summarizer function ---
def summarize_youtube(youtube_video_url):
    """
    Extract transcript using YouTube API (with Whisper fallback).
    Clean the transcript and summarize using Gemini.
    Returns (summary_text, error_message).
    If summary_text is not None -> success; if error_message is not None -> failure.
    """
    try:
        # Use the new enhanced transcript extraction with Whisper fallback
        print("\n📺 Extracting YouTube transcript (YouTube API + Whisper fallback)...")
        transcript_result = fetch_final_transcript(youtube_video_url)

        if transcript_result["error"]:
            return None, transcript_result["error"]

        transcript_text = transcript_result["transcript"]
        method_used = transcript_result["method_used"]

        if not transcript_text:
            return None, "No transcript content obtained"

        print(f"✅ Transcript obtained via {method_used.upper()}")
        print(f"📊 Transcript length: {len(transcript_text)} characters")

        # Summarize with Gemini
        prompt = (
            "You are a YouTube video summarizer. You will be taking the transcript text\n"
            "and summarizing the entire video, providing the important summary in points\n"
            "within 250 words. Please provide the summary of the text given here: "
        )

        summary = generate_with_gemini(prompt + transcript_text, model="gemini-2.5-flash")
        
        if not summary or "failed" in summary.lower():
            return None, "Gemini summarization failed"

        return summary, None

    except Exception as e:
        return None, str(e)


# ============================================================================
# MANUAL VIDEO FILE UPLOAD SUPPORT
# ============================================================================

def extract_audio_from_video(video_file_path: str, temp_dir: str = None) -> Optional[str]:
    """
    Extract audio from a locally uploaded video file using FFmpeg.
    Supports MP4, MOV, AVI, MKV, WebM, and other common video formats.
    
    Args:
        video_file_path: Path to the video file
        temp_dir: Temporary directory to store extracted audio
    
    Returns:
        Path to extracted audio file or None if failed
    """
    if not FFMPEG_AVAILABLE:
        print("❌ FFmpeg not available. Install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html or use: choco install ffmpeg")
        print("   Mac: brew install ffmpeg")
        print("   Linux: sudo apt-get install ffmpeg")
        return None
    
    if not os.path.exists(video_file_path):
        print(f"❌ Video file not found: {video_file_path}")
        return None
    
    try:
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        # Create output audio file path
        video_name = os.path.splitext(os.path.basename(video_file_path))[0]
        output_audio_path = os.path.join(temp_dir, f"extracted_audio_{video_name}.{AUDIO_FORMAT}")
        
        print(f"🎬 Extracting audio from video: {os.path.basename(video_file_path)}")
        
        # Use FFmpeg to extract audio
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_file_path,
            '-q:a', '9',  # Audio quality (lower=better, 0-9)
            '-n',  # Do not overwrite output files
            output_audio_path
        ]
        
        # Run FFmpeg (suppress output for cleaner console)
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0 and 'already exists' not in result.stderr:
            print(f"❌ FFmpeg error: {result.stderr}")
            return None
        
        if os.path.exists(output_audio_path):
            file_size_mb = os.path.getsize(output_audio_path) / (1024 * 1024)
            print(f"✅ Audio extracted successfully ({file_size_mb:.2f} MB)")
            return output_audio_path
        else:
            print("❌ Audio extraction failed")
            return None
    
    except Exception as e:
        print(f"❌ Error extracting audio: {str(e)}")
        return None


def summarize_uploaded_video(video_file_path: str) -> tuple:
    """
    Complete pipeline for uploading and summarizing a local video file.
    
    Workflow:
    1. Extract audio from video file using FFmpeg
    2. Transcribe audio using Whisper
    3. Summarize transcript using Gemini
    4. Clean up temporary files
    
    Args:
        video_file_path: Path to the local video file
    
    Returns:
        Tuple of (summary_text, error_message)
        If summary_text is not None -> success
        If error_message is not None -> failure
    """
    audio_path = None
    
    try:
        print(f"\n{'='*70}")
        print(f"📹 Processing uploaded video: {os.path.basename(video_file_path)}")
        print(f"{'='*70}\n")
        
        # Step 1: Extract audio from video
        print("Step 1️⃣ : Extracting audio from video...")
        audio_path = extract_audio_from_video(video_file_path)
        if not audio_path:
            return None, "Failed to extract audio from video"
        
        # Step 2: Generate transcript from audio using Whisper
        print("\nStep 2️⃣ : Transcribing audio using Whisper...")
        transcript_text = generate_transcript_from_audio(audio_path)
        if not transcript_text:
            return None, "Failed to transcribe audio"
        
        print(f"✅ Transcription complete ({len(transcript_text)} characters)")
        
        # Step 3: Summarize with Gemini
        print("\nStep 3️⃣ : Generating summary using Gemini...")
        prompt = (
            "You are a video summarizer. You will be taking the transcript text\n"
            "and summarizing the entire video, providing the important summary in points\n"
            "within 250 words. Please provide the summary of the text given here:\n\n"
        )
        
        summary = generate_with_gemini(prompt + transcript_text, model="gemini-2.5-flash")
        
        if not summary or "failed" in summary.lower():
            return None, "Gemini summarization failed"
        
        print(f"\n✅ SUCCESS: Video summarized!")
        return summary, None
        
    except Exception as e:
        return None, str(e)
    
    finally:
        # Always clean up temporary audio file
        if audio_path:
            cleanup_audio_file(audio_path)


def prompt_for_video_file() -> Optional[str]:
    """
    Prompt user to enter path to a video file.
    Validates that the file exists and has a supported video extension.
    
    Returns:
        Path to video file or None if invalid
    """
    supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    
    print("\n📹 Supported video formats:", ", ".join(supported_formats))
    print("   (Windows tip: Drag and drop the video file into terminal to get full path)")
    
    while True:
        video_path = input("Enter video file path (or 'cancel' to go back): ").strip()
        
        if video_path.lower() == 'cancel':
            return None
        
        # Remove quotes if user copy-pasted with quotes
        video_path = video_path.strip('"\'')
        
        if not os.path.exists(video_path):
            print("❌ File not found. Please check the path and try again.")
            continue
        
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in supported_formats:
            print(f"❌ Unsupported format. Supported formats: {', '.join(supported_formats)}")
            continue
        
        return video_path


# Orchestration / Main Loop
def main():
    print("Starting Chatbot with persistent ChromaDB and Enhanced YouTube Transcript Extraction...")
    store = ChromaMemoryStore()
    # create session file for this run
    session_file = new_session_file()
    print("Session file:", session_file)

    print("\nReady. Type text, type 'voice' to speak, 'summarize' to summarize a YouTube video, 'upload' to summarize a local video file, 'exit' to quit.")
    print("Note: YouTube summarization now supports Whisper fallback for videos without captions!\n")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if user_input.lower() == "voice":
            try:
                user_text = listen_from_mic()
                print("You (voice):", user_text)
            except Exception as e:
                print("Voice error:", e)
                continue
        elif user_input.lower().startswith("summarize") or user_input.lower() == "summarize":
            # Ask for YouTube URL (either inline after command or via prompt)
            # Allow "summarize <url>" or separate prompt
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                youtube_link = parts[1].strip()
            else:
                youtube_link = input("Enter YouTube video link to summarize: ").strip()
            if not youtube_link:
                print("No YouTube link provided.")
                continue

            print("Fetching transcript and summarizing... (this may take a moment)")
            summary, err = summarize_youtube(youtube_link)
            if err:
                print("Summarizer error:", err)
                continue
            if not summary:
                print("Summarizer returned no content.")
                continue

            # Save user request and assistant summary to session & Chroma
            append_to_session(session_file, "user", f"summarize: {youtube_link}")
            append_to_session(session_file, "assistant", summary)
            try:
                store.add_text(f"Assistant (summary): {summary}", source_file=session_file, metadata={"role": "assistant", "type": "summary"})
            except Exception as e:
                print("Error adding summary to Chroma:", e)

            print("\nSummary:\n")
            print(summary)
            try:
                speak_text(summary)
            except Exception:
                pass

            continue  # skip normal chat flow for summarize command

        elif user_input.lower() == "upload":
            video_path = prompt_for_video_file()
            if not video_path:
                print("No video file provided.")
                continue

            print("Processing uploaded video and summarizing... (this may take a moment)")
            summary, err = summarize_uploaded_video(video_path)
            if err:
                print("Summarizer error:", err)
                continue
            if not summary:
                print("Summarizer returned no content.")
                continue

            # Save user request and assistant summary to session & Chroma
            append_to_session(session_file, "user", f"upload: {video_path}")
            append_to_session(session_file, "assistant", summary)
            try:
                store.add_text(f"Assistant (summary): {summary}", source_file=session_file, metadata={"role": "assistant", "type": "summary"})
            except Exception as e:
                print("Error adding summary to Chroma:", e)

            print("\nSummary:\n")
            print(summary)
            try:
                speak_text(summary)
            except Exception:
                pass

            continue  # skip normal chat flow for upload command

        else:
            user_text = user_input

        # Save to session .txt
        append_to_session(session_file, "user", user_text)

        # Add to Chroma (only new item)
        try:
            store.add_text(f"User: {user_text}", source_file=session_file, metadata={"role": "user"})
        except Exception as e:
            print("Error adding to Chroma:", e)

        # Build context from top-k similar memories (exclude the just-added item by score if needed)
        hits = store.query(user_text, top_k=TOP_K)
        if hits:
            context_lines = []
            for h in hits:
                # show saved text and source
                md = h.get("metadata", {}) or {}
                src = md.get("source", "")
                context_lines.append(f"- {h['text']} (src={src})")
            context = "\n".join(context_lines)
        else:
            context = "No relevant memories found."

        system_prompt = (
            "You are an assistant that uses the following memories to answer the user's query concisely.\n"
            f"Memories:\n{context}\n\nUser query: {user_text}\n\nAnswer concisely."
        )

        # Generate reply with Gemini
        reply = generate_with_gemini(system_prompt)
        if not reply:
            reply = "Sorry, I couldn't generate a reply."

        # Save assistant reply to session file & Chroma
        append_to_session(session_file, "assistant", reply)
        try:
            store.add_text(f"Assistant: {reply}", source_file=session_file, metadata={"role": "assistant"})
        except Exception as e:
            print("Error adding assistant reply to Chroma:", e)

        # Print & speak
        print("\nAssistant:", reply)
        try:
            speak_text(reply)
        except Exception:
            pass

if __name__ == "__main__":
    main()