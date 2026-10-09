"""
Centralized Configuration for IndiaWalls AI Calling System.
Loads environment variables from .env and exposes strongly typed settings.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Ensure environment is loaded from project root and backend folder
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

# Load backend/.env first, then root .env (or vice versa with override)
if (BACKEND_DIR / ".env").exists():
    load_dotenv(dotenv_path=BACKEND_DIR / ".env", override=True)
if (ROOT_DIR / ".env").exists():
    load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)
load_dotenv(override=False)

# Server & Network Settings
WS_HOST = os.environ.get("WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("WS_PORT", "8765"))

# LLM Backend: "gemini" (Google GenAI) or "groq" (Groq Cloud)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "gemini").lower()

# Gemini Settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_API_KEYS = os.environ.get("GEMINI_API_KEYS", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Groq Settings (Fallback)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "qwen/qwen3.8-27b")

# Kokoro TTS Engine: "inproc" (Direct inside server) or "remote" (HTTP standalone)
KOKORO_MODE = os.environ.get("KOKORO_MODE", "inproc").lower()
KOKORO_TTS_URL = os.environ.get("KOKORO_TTS_URL", "http://localhost:8880").rstrip("/")
KOKORO_TTS_VOICE = os.environ.get("KOKORO_TTS_VOICE", "hf_alpha")
KOKORO_TTS_SPEED = float(os.environ.get("KOKORO_TTS_SPEED", "1.35"))

# Speech-to-Text (STT): "local" (IndicConformer CTC) or "groq" (Whisper)
STT_BACKEND = os.environ.get("STT_BACKEND", "local").lower()
SHERPA_MODEL_DIR = str(ROOT_DIR / os.environ.get("SHERPA_MODEL_DIR", "models/indicconformer-hi"))
SHERPA_NUM_THREADS = int(os.environ.get("SHERPA_NUM_THREADS", "2"))

# Supabase Cloud Database Settings (Primary & Exclusive Database)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

# Hardcoded Admin Login Credentials (Username & Password)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin").strip()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123").strip()
JWT_SECRET = os.environ.get("JWT_SECRET", "indiawalls-secure-auth-secret-key").strip()

