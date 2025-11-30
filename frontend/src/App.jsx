import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Mic, Youtube, Sparkles, Menu, Plus, Trash2, MessageSquare } from 'lucide-react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const recognitionRef = useRef(null);

  // Initialize Web Speech API
  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setIsListening(false);
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, []);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Load messages when session changes
  useEffect(() => {
    if (currentSessionId) {
      loadSession(currentSessionId);
    }
  }, [currentSessionId]);

  // Auto-scroll to bottom with smooth behavior
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages]);

  const loadSessions = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/sessions`);
      setSessions(response.data);
      if (!currentSessionId && response.data.length > 0) {
        setCurrentSessionId(response.data[0].session_id);
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
    }
  };

  const loadSession = async (sessionId) => {
    try {
      const response = await axios.get(`${API_BASE}/api/session/${sessionId}`);
      setMessages(response.data.messages || []);
      // Scroll to top first, then to bottom after messages load
      setTimeout(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        }
      }, 100);
    } catch (error) {
      console.error('Error loading session:', error);
    }
  };

  const createNewSession = async () => {
    try {
      const response = await axios.post(`${API_BASE}/api/session/new`);
      const newSessionId = response.data.session_id;
      await loadSessions();
      setCurrentSessionId(newSessionId);
      setMessages([]);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      await axios.delete(`${API_BASE}/api/session/${sessionId}`);
      await loadSessions();
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  };

  const sendMessage = async (messageText = input) => {
    if (!messageText.trim() || loading) return;

    if (!currentSessionId) {
      await createNewSession();
      // Wait a bit for session to be created
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    const userMessage = { role: 'user', content: messageText };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/api/chat`, {
        message: messageText,
        session_id: currentSessionId
      });

      const assistantMessage = { role: 'assistant', content: response.data.reply };
      setMessages(prev => [...prev, assistantMessage]);
      await loadSessions(); // Refresh to get updated title
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please check if the backend is running and your API key is valid.' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleYoutubeSummary = async () => {
    const url = prompt('Enter YouTube video URL:');
    if (!url) return;

    if (!currentSessionId) {
      await createNewSession();
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    const userMessage = { role: 'user', content: `Summarize: ${url}` };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/api/summarize`, {
        url,
        session_id: currentSessionId
      });

      const summaryMessage = { role: 'assistant', content: response.data.summary };
      setMessages(prev => [...prev, summaryMessage]);
      await loadSessions();
    } catch (error) {
      console.error('Error summarizing video:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I could not summarize the video. Make sure it has English captions.' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const startVoiceInput = () => {
    if (recognitionRef.current && !isListening) {
      setIsListening(true);
      recognitionRef.current.start();
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage();
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            className="sidebar"
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          >
            <div className="sidebar-header">
              <h1 className="sidebar-title">
                <Sparkles size={24} />
                Memory Assistant
              </h1>
              <button className="new-chat-btn" onClick={createNewSession}>
                <Plus size={20} />
                New Chat
              </button>
            </div>

            <div className="sessions-list">
              {sessions.map((session) => (
                <motion.div
                  key={session.session_id}
                  className={`session-item ${currentSessionId === session.session_id ? 'active' : ''}`}
                  onClick={() => setCurrentSessionId(session.session_id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <MessageSquare size={16} />
                  <span className="session-title">{session.title}</span>
                  <button
                    className="delete-session-btn"
                    onClick={(e) => deleteSession(session.session_id, e)}
                  >
                    <Trash2 size={14} />
                  </button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="main-content">
        <div className="chat-header">
          <button className="menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <Menu size={24} />
          </button>
          <h2 className="chat-title">
            {sessions.find(s => s.session_id === currentSessionId)?.title || 'New Chat'}
          </h2>
        </div>

        <div className="messages-container" ref={messagesContainerRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <Sparkles size={64} className="empty-icon" />
              <h2>Start a Conversation</h2>
              <p>Ask me anything, summarize YouTube videos, or use voice input!</p>
            </div>
          ) : (
            <AnimatePresence>
              {messages.map((msg, idx) => (
                <motion.div
                  key={idx}
                  className={`message ${msg.role}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="message-content">
                    <div className="message-avatar">
                      {msg.role === 'user' ? '👤' : '🤖'}
                    </div>
                    <div className="message-text">{msg.content}</div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
          
          {loading && (
            <motion.div
              className="message assistant loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="message-content">
                <div className="message-avatar">🤖</div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <form onSubmit={handleSubmit} className="input-form">
            <button
              type="button"
              className="icon-btn youtube-btn"
              onClick={handleYoutubeSummary}
              title="Summarize YouTube Video"
            >
              <Youtube size={20} />
            </button>

            <button
              type="button"
              className={`icon-btn mic-btn ${isListening ? 'listening' : ''}`}
              onClick={startVoiceInput}
              title="Voice Input"
            >
              <Mic size={20} />
            </button>

            <input
              type="text"
              className="message-input"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
            />

            <button
              type="submit"
              className="send-btn"
              disabled={!input.trim() || loading}
            >
              <Send size={20} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
