import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Brain, Sparkles, MessageSquare, Zap, Shield, Cpu, ArrowRight, Github, Star } from 'lucide-react';
import './HomePage.css';

export default function HomePage({ onEnterChat }) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const features = [
    {
      icon: <Brain size={32} />,
      title: "AI-Powered Memory",
      description: "Never lose context! ChromaDB stores every conversation as vector embeddings, enabling intelligent semantic search across all your chat history."
    },
    {
      icon: <MessageSquare size={32} />,
      title: "Intelligent Conversations",
      description: "Powered by Google Gemini 2.0 Flash for lightning-fast responses that understand context and remember what you've discussed."
    },
    {
      icon: <Zap size={32} />,
      title: "Real-time Learning",
      description: "The AI continuously learns from your conversations, becoming more personalized and helpful with every interaction."
    },
    {
      icon: <Shield size={32} />,
      title: "Privacy First",
      description: "Your conversations are stored locally on your machine. No cloud storage, complete privacy and data ownership."
    },
    {
      icon: <Cpu size={32} />,
      title: "Advanced Vector Search",
      description: "Uses sentence transformers and ChromaDB for state-of-the-art semantic similarity search across your memories."
    },
    {
      icon: <Sparkles size={32} />,
      title: "Smart Organization",
      description: "Auto-generated chat titles, persistent sessions, and organized history make it easy to find past conversations."
    }
  ];

  const stats = [
    { number: "99.9%", label: "Uptime" },
    { number: "< 100ms", label: "Response Time" },
    { number: "∞", label: "Memory Capacity" },
    { number: "24/7", label: "Availability" }
  ];

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="homepage">
      {/* Animated Background */}
      <div className="background-wrapper">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
        <div className="grid-overlay"></div>
        
        {/* Mouse follower effect */}
        <motion.div
          className="mouse-glow"
          animate={{
            x: mousePosition.x - 250,
            y: mousePosition.y - 250,
          }}
          transition={{ type: 'spring', damping: 30, stiffness: 200 }}
        />
      </div>

      {/* Navigation */}
      <motion.nav 
        className="navbar"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <div className="nav-container">
          <div className="logo">
            <Brain className="logo-icon" />
            <span className="logo-text">Memory Assistant</span>
          </div>
          <div className="nav-links">
            <a href="#features" onClick={(e) => { e.preventDefault(); scrollToSection('features'); }}>Features</a>
            <a href="#about" onClick={(e) => { e.preventDefault(); scrollToSection('about'); }}>About</a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">
              <Github size={20} />
            </a>
          </div>
        </div>
      </motion.nav>

      {/* Hero Section */}
      <section className="hero-section">
        <motion.div
          className="hero-content"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          <motion.div
            className="badge"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Star size={16} />
            <span>Powered by AI & ChromaDB</span>
          </motion.div>

          <motion.h1
            className="hero-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            Personalized Memory
            <br />
            <span className="gradient-text">Assistant Powered by AI</span>
          </motion.h1>

          <motion.p
            className="hero-subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            Your intelligent companion that remembers everything. Built with ChromaDB vector database 
            and Google Gemini AI, this assistant learns from every conversation and provides 
            context-aware responses based on your entire chat history.
          </motion.p>

          <motion.div
            className="hero-buttons"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1 }}
          >
            <button className="primary-btn" onClick={onEnterChat}>
              <span>Start Chatting</span>
              <ArrowRight size={20} />
            </button>
            <button className="secondary-btn" onClick={() => scrollToSection('about')}>
              <span>Learn More</span>
            </button>
          </motion.div>

          {/* Stats */}
          <motion.div
            className="stats-grid"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.2 }}
          >
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                className="stat-item"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 1.4 + index * 0.1 }}
              >
                <div className="stat-number">{stat.number}</div>
                <div className="stat-label">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>

        {/* 3D Visual Element */}
        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          <div className="floating-card card-1">
            <Brain size={40} />
            <span>Neural Network</span>
          </div>
          <div className="floating-card card-2">
            <Cpu size={40} />
            <span>Vector DB</span>
          </div>
          <div className="floating-card card-3">
            <Sparkles size={40} />
            <span>AI Learning</span>
          </div>
          <div className="center-orb">
            <div className="orb-ring"></div>
            <div className="orb-ring"></div>
            <div className="orb-ring"></div>
            <Zap size={48} />
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="features-section" id="features">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="section-title">Powerful Features</h2>
          <p className="section-subtitle">
            Built with cutting-edge technology for an unparalleled AI experience
          </p>
        </motion.div>

        <div className="features-grid">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              className="feature-card"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -10, scale: 1.02 }}
            >
              <div className="feature-icon">{feature.icon}</div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* About/Benefits Section */}
      <section className="benefits-section" id="about">
        <motion.div
          className="benefits-container"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="section-title">How It Helps You</h2>
          <div className="benefits-grid">
            <div className="benefit-item">
              <div className="benefit-number">01</div>
              <h3>Never Forget Important Details</h3>
              <p>Every conversation is stored with semantic embeddings, allowing the AI to recall relevant information from past discussions automatically.</p>
            </div>
            <div className="benefit-item">
              <div className="benefit-number">02</div>
              <h3>Context-Aware Assistance</h3>
              <p>Unlike traditional chatbots, our AI remembers your preferences, past questions, and builds upon previous conversations for truly personalized help.</p>
            </div>
            <div className="benefit-item">
              <div className="benefit-number">03</div>
              <h3>Knowledge Base Management</h3>
              <p>Perfect for students, researchers, and professionals who need to manage and query large amounts of information efficiently.</p>
            </div>
            <div className="benefit-item">
              <div className="benefit-number">04</div>
              <h3>YouTube Video Summarization</h3>
              <p>Extract key insights from YouTube videos instantly with AI-powered transcript analysis and summarization.</p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Tech Stack Section */}
      <section className="tech-section" id="about">
        <motion.div
          className="tech-container"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="section-title">Built With Modern Tech</h2>
          <div className="tech-stack">
            <div className="tech-item">
              <div className="tech-logo">🧠</div>
              <span>Google Gemini 2.0</span>
            </div>
            <div className="tech-item">
              <div className="tech-logo">🗄️</div>
              <span>ChromaDB</span>
            </div>
            <div className="tech-item">
              <div className="tech-logo">⚛️</div>
              <span>React 18</span>
            </div>
            <div className="tech-item">
              <div className="tech-logo">⚡</div>
              <span>FastAPI</span>
            </div>
            <div className="tech-item">
              <div className="tech-logo">🤖</div>
              <span>Sentence Transformers</span>
            </div>
            <div className="tech-item">
              <div className="tech-logo">🔮</div>
              <span>Framer Motion</span>
            </div>
          </div>
        </motion.div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <motion.div
          className="cta-content"
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="cta-title">Ready to Experience AI Memory?</h2>
          <p className="cta-subtitle">Start your first conversation and watch the magic happen</p>
          <button className="cta-button" onClick={onEnterChat}>
            <span>Launch Chat Interface</span>
            <ArrowRight size={24} />
          </button>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>© 2025 Personalized Memory Assistant. Powered by AI.</p>
        <div className="footer-links">
          <a href="https://github.com">GitHub</a>
          <span>•</span>
          <a href="#">Privacy</a>
          <span>•</span>
          <a href="#">Terms</a>
        </div>
      </footer>
    </div>
  );
}

