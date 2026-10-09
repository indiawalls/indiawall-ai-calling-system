# IndiaWalls AI Voice Calling System — Production Architecture

This project is a high-performance, real-time Speech-to-Speech (S2S) voice calling assistant for **Indiawalls Infratech Private Limited** supporting both PSTN phone calls (via **Exotel**) and Web microphone calls (via **Pipecat** and **WebSockets**).

---

## 📁 Project Directory Structure

```text
tipsg-calling-ai/
│
├── backend/                             # Core Python Application (Production Modular Layout)
│   ├── app.py                           # FastAPI application factory, lifespan pre-warming & CORS
│   ├── config.py                        # Centralized settings & environment loader (.env)
│   ├── run.py                           # CLI entrypoint to start the server (`python backend/run.py`)
│   │
│   ├── database/                        # Dual Persistence Layer (Supabase Cloud + Local SQLite Fallback)
│   │   ├── connection.py                # Supabase client initializer and SQLite connection pool
│   │   ├── repository.py                # Unified repository CRUD (calls, messages, turn counts, stats)
│   │   └── schema.sql                   # Supabase PostgreSQL schema with tables, indices & RLS
│   │
│   ├── services/                        # AI Model & Speech Pipelines
│   │   ├── llm/                         # Language Model Services
│   │   │   ├── gemini_rotator.py        # Multi-Account API key rotator (429 automatic failover & cooldown)
│   │   │   └── client_factory.py        # Dynamic client builder for Google Gemini & Groq
│   │   ├── stt/                         # Speech-to-Text Services
│   │   │   └── sherpa_service.py        # Offline IndicConformer NeMo CTC (Sherpa-ONNX streaming singleton)
│   │   └── tts/                         # Text-to-Speech Services
│   │       ├── kokoro_service.py        # In-process Kokoro-82M streaming audio synthesizer
│   │       └── normalizer.py            # Spoken Hindi phone digit and text normalizer
│   │
│   ├── telephony/                       # Audio Transport & Protocols
│   │   ├── exotel_serializer.py         # Exotel 16-bit Linear PCM stream serializer & caller ID extraction
│   │   ├── echo_filter.py               # Phone acoustic speaker echo suppressor
│   │   └── raw_serializer.py            # Browser raw PCM audio serializer
│   │
│   ├── knowledge/                       # Knowledge Base & Prompts
│   │   ├── prompts.py                   # IndiaWalls system prompt ('Priya') & interactive presets
│   │   └── rag_kb.md                    # Official IndiaWalls product catalog & firm background
│   │
│   └── api/                             # REST & WebSocket API Routers
│       ├── calls.py                     # /api/calls, /api/calls/{id}, /api/stats, /api/calls/clear
│       ├── system.py                    # /api/health, /api/voices, /api/firm-info, /api/test-pipeline
│       └── websockets.py                # WebSocket endpoints: /ws (Web) & /exotel-ws (Exotel PSTN)
├── assets/                              # Static Audio & Branding Assets
│   └── audio/                           # IndiaWalls greeting audio (MP3, 8k, 16k, 24k WAV)
│
├── index.html                           # Real-time Testing Dashboard, Latency Profiler & Call History
│
├── models/                              # Offline ONNX model weights
│   └── indicconformer-hi/               # model.int8.onnx and tokens.txt
│
├── scripts/                             # Utility & Maintenance Scripts
│   ├── download_models.py               # Download IndicConformer model weights from HuggingFace
│   ├── benchmark_latency.py             # Pipeline stage latency profiler
│   └── test_supabase.py                 # Supabase credentials & schema verification script
│
├── tests/                               # Test Cases & Test Audio
│   └── test_audio/                      # Sample telephony WAV/MPEG audio files for latency profiling
│
├── server.py                            # Backward-compatible root entrypoint (`python server.py`)
├── requirements.txt                     # Production pip dependencies
├── .env                                 # Local API keys and credentials (ignored in git)
└── .env.example                         # Documented template for developer setup
```

---

## 🚀 How to Run the Project

### 1. Start the Voice Server
You can launch using either command:
```bash
python backend/run.py
```
*or (backward compatible):*
```bash
python server.py
```

Open your browser to:
👉 **`http://localhost:8765/`**

---

## 🗄️ Setting Up Supabase Database

1. Open your [Supabase Dashboard](https://supabase.com/dashboard).
2. Go to the **SQL Editor** tab.
3. Open [`backend/database/schema.sql`](file:///c:/Machine%20Learning/tipsg-calling-ai/backend/database/schema.sql) and paste the SQL script to create the `calls` and `call_messages` tables.
4. In [`.env`](file:///c:/Machine%20Learning/tipsg-calling-ai/.env), provide your project URL and service/anon key:
   ```env
   SUPABASE_URL=https://xyzcompany.supabase.co
   SUPABASE_KEY=eyJhbGciOi...
   ```
5. Run the verification script:
   ```bash
   python scripts/test_supabase.py
   ```
*If `SUPABASE_URL` is left blank, the system automatically runs on local SQLite (`calls.db`) with zero extra configuration!*

---

## 🔑 Multi-Account Gemini Key Rotation

To bypass Google AI Studio's 15 RPM / 1,500 RPD free tier limit, add multiple API keys separated by commas in [`.env`](file:///c:/Machine%20Learning/tipsg-calling-ai/.env):
```env
GEMINI_API_KEYS=key_from_acc_1,key_from_acc_2,key_from_acc_3
```
- Calls will be distributed evenly across accounts in a **Round-Robin** sequence.
- If any key receives a `429 ResourceExhausted` rate-limit error, it is placed on a **60-second cooldown** and requests automatically shift to the next active key.

---

## 📞 Endpoints Summary

| Route | Protocol | Description |
|---|---|---|
| `/` | HTTP GET | Serves the Testing Console & Call History Dashboard (`index.html`) |
| `/ws` | WebSocket | Real-time browser microphone voice calling |
| `/exotel-ws` | WebSocket | Real-time Exotel telephony phone stream (16kHz PCM) |
| `/exotel-call` | HTTP GET/POST | Exotel IVR Applet Webhook (returns XML connecting to `/exotel-ws`) |
| `/api/calls` | HTTP GET | Fetches paginated call logs with durations, turns, and phone numbers |
| `/api/calls/{id}` | HTTP GET | Fetches full conversation dialogue transcript for a call |
| `/api/stats` | HTTP GET | Returns KPI counts and Gemini key rotator diagnostics |
| `/api/health` | HTTP GET | System health check (STT, LLM, TTS, DB status) |
| `/api/test-pipeline`| HTTP POST | Single-turn query test with per-stage latency breakdown |
