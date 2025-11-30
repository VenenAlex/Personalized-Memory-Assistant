# Personalized Memory Assistant - ChatGPT-like Interface

## 🚀 Quick Start

### Backend Setup:
1. Open a terminal and navigate to the backend folder:
   ```
   cd backend
   ```

2. Install Python dependencies (if not already installed):
   ```
   pip install fastapi uvicorn python-multipart chromadb sentence-transformers google-generativeai python-dotenv yt-dlp
   ```

3. Start the backend server:
   ```
   python server.py
   ```
   Backend will run at: http://localhost:8000

### Frontend Setup:
1. Open a **new terminal** and navigate to the frontend folder:
   ```
   cd frontend
   ```

2. Install dependencies (first time only):
   ```
   npm install
   ```

3. Start the React app:
   ```
   npm start
   ```
   Frontend will run at: http://localhost:3000

## ✨ Features

- **ChatGPT-like Interface**: Modern, beautiful Glass UI design with animations
- **Session Management**: Smart session naming using AI (like ChatGPT)
- **Voice Input**: Click the microphone icon to speak your message
- **YouTube Summarizer**: Click the YouTube icon to summarize any video
- **Persistent Memory**: ChromaDB stores all conversations for context-aware responses
- **Beautiful Animations**: Smooth transitions and loading states

## 🎨 Interface Features

- **Sidebar**: View all chat sessions with AI-generated titles
- **New Chat**: Create new conversations with one click
- **Delete Sessions**: Remove old conversations easily
- **Glass Morphism**: Modern frosted glass design
- **Responsive**: Works on desktop and mobile
- **Auto-scroll**: Messages automatically scroll to bottom
- **Typing Indicator**: Shows when AI is thinking

## 🎯 How to Use

1. **Start Chatting**: Type your message and press Enter or click Send
2. **Voice Input**: Click the microphone icon and speak
3. **YouTube Summary**: Click the YouTube icon, paste a video URL
4. **Switch Sessions**: Click any session in the sidebar to switch
5. **New Chat**: Click "New Chat" button to start fresh

## 📁 Project Structure

```
backend/
  server.py          # FastAPI backend with AI session naming
frontend/
  src/
    App.jsx          # Main React component
    App.css          # Glass UI styles with animations
    index.js         # React entry point
  public/
    index.html       # HTML template
sessions/            # Stores all chat sessions (JSON format)
chroma_db/           # ChromaDB vector database
```

## 🔑 Environment Variables

Make sure your `.env` file has:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## 🎉 Enjoy Your Modern ChatGPT Clone!
