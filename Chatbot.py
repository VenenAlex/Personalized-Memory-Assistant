#!/usr/bin/env python3
"""
chat.py
Voice+Text chatbot with per-session .txt files and persistent ChromaDB memory.
Saves every user message to a session file and adds only new messages to ChromaDB.
Generates replies with Google Gemini (google-generativeai). Works with both
old (text.generate) and newer (chat.completions.create) client APIs.
"""
import re
import os
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime, timezone

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# --------- Configuration ---------
SESSIONS_DIR = Path("sessions")
CHROMA_PERSIST_DIR = Path("chroma_db")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
GEMINI_MODEL = "gemini-2.0-flash" 

import chromadb
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
import pyttsx3

try:
    from yt_dlp import YoutubeDL
except Exception:
    YoutubeDL = None

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


#Voice & TTS helpers
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

#Session file helpers
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

# --- YouTube summarizer function (merged from summarizer.py) ---
def summarize_youtube(youtube_video_url):
    """
    Extract English subtitles (auto or manual) using yt_dlp, clean the VTT file,
    and summarize the transcript using Gemini. Returns (summary_text, error_message).
    If summary_text is not None -> success; if error_message is not None -> failure.
    """
    if YoutubeDL is None:
        return None, "yt_dlp not installed. Install yt-dlp to enable summarizer."

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
            info = ydl.extract_info(youtube_video_url, download=True)
            video_id = info.get('id', None)

        # Subtitle filename produced by yt_dlp (common pattern)
        transcript_file = f"{video_id}.en.vtt"
        if not os.path.exists(transcript_file):
            return None, "Could not find an English transcript for this video."

        # Read and clean the transcript text
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        transcript_text = ""
        for line in lines:
            if '-->' not in line and not line.strip().isdigit() and line.strip() and "WEBVTT" not in line:
                cleaned_line = re.sub(r'<[^>]+>', '', line).strip()
                transcript_text += " " + cleaned_line

        # Clean up: delete the temporary subtitle file
        try:
            os.remove(transcript_file)
        except Exception:
            pass

        prompt = (
            "You are a YouTube video summarizer. You will be taking the transcript text\n"
            "and summarizing the entire video, providing the important summary in points\n"
            "within 250 words. Please provide the summary of the text given here: "
        )

        # Use a newer Gemini model if available
        summary = generate_with_gemini(prompt + transcript_text, model="gemini-2.5-flash")
        return summary, None

    except Exception as e:
        return None, str(e)
    


#Orchestration / Main Loop
def main():
    print("Starting Chatbot with persistent ChromaDB....")
    store = ChromaMemoryStore()
    # create session file for this run
    session_file = new_session_file()
    print("Session file:", session_file)

    print("Ready. Type text, type 'voice' to speak, 'summarize' to summarize a YouTube video, 'exit' to quit.")

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