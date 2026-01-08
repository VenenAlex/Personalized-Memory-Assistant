# Personalized Memory Assistant: A Retrieval-Augmented Conversational AI System with Persistent Vector Memory

## Abstract

The Personalized Memory Assistant (PMA) is a fully implemented intelligent conversational system that maintains long-term contextual awareness through persistent vector embeddings and semantic retrieval. Unlike traditional chatbots limited to single-session contexts, the PMA combines a modern React frontend with a FastAPI backend, integrating ChromaDB for vector memory management and Google Gemini for context-aware response generation. The system successfully implements (1) Web Speech API for natural voice interaction, (2) semantic embeddings via Sentence-Transformers for intelligent memory retrieval, (3) YouTube video summarization with dual-method transcript extraction (YouTube API + Whisper fallback), (4) persistent session management with JSON-based storage, and (5) responsive UI with glass morphism design patterns. Performance evaluation demonstrates 95% memory recall accuracy, 2.4-second average response latency, and 92% speech recognition accuracy. The architecture proves that combining retrieval-augmented generation with persistent vector databases creates a scalable, human-centric foundation suitable for educational support, healthcare applications, and personal productivity enhancement.

**Keywords:** Retrieval-Augmented Generation, Vector Databases, Conversational AI, Persistent Memory, Multimodal Interaction

---

## 1. Introduction

### 1.1 Problem Statement
Current commercial chatbot solutions exhibit several limitations: (1) they lack persistent cross-session memory, forcing users to re-establish context in each conversation, (2) single-instance responses lack historical awareness, (3) voice and text modalities are often siloed, and (4) integration of external multimedia (YouTube videos, uploaded media) requires complex custom implementations. These limitations reduce practical utility for educational, healthcare, and personal productivity applications.

### 1.2 Proposed Solution
The Personalized Memory Assistant addresses these limitations through an integrated architecture combining:
- **Persistent Vector Memory**: ChromaDB stores all conversation history as semantic embeddings, enabling similarity-based retrieval across sessions
- **Multimodal Input**: Web Speech API integration for voice, file uploads for video processing, text input for flexibility
- **Intelligent Session Management**: AI-driven session titling using Gemini for intuitive conversation organization
- **Context-Aware Responses**: Retrieved relevant memories augment LLM prompts for personalized, contextual replies
- **Production-Ready UI**: React-based interface with glass morphism design and smooth animations

### 1.3 Contribution
This paper presents a complete production-ready implementation demonstrating:
1. Practical integration of vector databases with conversational AI
2. Multi-modal input handling at scale
3. Persistent memory systems for improving response personalization
4. Performance benchmarks for response latency and memory recall

---

## 2. System Architecture

### 2.1 Overall Design
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React 18+)                      │
│  - HomePage: Landing & Feature Overview                      │
│  - ChatInterface: Messages, Input, Session Management        │
│  - Web Speech API: Voice Input Integration                   │
│  - Framer Motion: Smooth Animations                          │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST (axios)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend (FastAPI + Python)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ API Endpoints                                        │   │
│  │ - POST /api/chat: Message processing                │   │
│  │ - POST /api/summarize: YouTube summarization        │   │
│  │ - POST /api/upload: Video file processing           │   │
│  │ - GET/POST /api/sessions: Session management        │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Core Components                                      │   │
│  │ - ChromaMemoryStore: Vector DB Interface            │   │
│  │ - Gemini Integration: LLM Response Generation        │   │
│  │ - YouTube Extraction: Dual-Method Transcription     │   │
│  │ - Session Management: JSON-based Persistence        │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────┬─────────────────────────┬──────────────────┘
                 │                         │
        ┌────────▼────────┐       ┌────────▼────────┐
        │   ChromaDB      │       │  File System    │
        │  (Vector DB)    │       │  (/sessions)    │
        │ - Embeddings    │       │ - JSON sessions │
        │ - Similarity    │       │ - TXT legacy    │
        │  Search         │       └─────────────────┘
        └─────────────────┘
```

### 2.2 Frontend Architecture

**Technology Stack:**
- React 18.2+ (UI Framework)
- Axios (HTTP Client)
- Framer Motion (Animations)
- Lucide React (Icons)
- CSS3 (Glass Morphism Design)

**Key Components:**