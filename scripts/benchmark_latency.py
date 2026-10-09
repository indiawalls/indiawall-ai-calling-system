import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
import kokoro
from kokoro import KPipeline
import torch

load_dotenv()

llm_backend = os.environ.get("LLM_BACKEND", "gemini").lower()
gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
gemini_key = os.environ.get("GEMINI_API_KEY", "")
groq_model = os.environ.get("GROQ_LLM_MODEL", "qwen/qwen3.8-27b")
groq_key = os.environ.get("GROQ_API_KEY", "")

# 1. Init TTS and preload voice
t_init = time.perf_counter()
pipe = KPipeline(lang_code='h')
voice_tensor = pipe.load_voice('hf_alpha')
print(f'TTS initialized in {(time.perf_counter()-t_init)*1000:.0f}ms')

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.knowledge import INDIAWALLS_SYSTEM_PROMPT

system_prompt = INDIAWALLS_SYSTEM_PROMPT
user_query = "आप कितने प्रकार की दीवारें बनाते हैं?"

print(f'--- SIMULATING REAL-TIME TURN (LLM: {llm_backend.upper()}) ---')
t_start = time.perf_counter()

# LLM call
t_llm_start = time.perf_counter()
active_model_name = ""

if llm_backend == "gemini" and gemini_key:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=gemini_key)
    active_model_name = f"Gemini ({gemini_model})"
    resp = client.models.generate_content(
        model=gemini_model,
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=80,
        ),
    )
    reply_text = resp.text.strip()
else:
    from groq import Groq
    client = Groq(api_key=groq_key)
    active_model_name = f"Groq ({groq_model})"
    resp = client.chat.completions.create(
        model=groq_model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ],
        max_completion_tokens=80,
        temperature=0.2
    )
    reply_text = resp.choices[0].message.content.strip()

t_llm_end = time.perf_counter()
llm_latency = (t_llm_end - t_llm_start) * 1000

# TTS synthesis (with preloaded voice)
t_tts_start = time.perf_counter()
first_audio_ms = None
total_audio_chunks = 0

with torch.inference_mode():
    for gs, ps, audio in pipe(reply_text, voice=voice_tensor, speed=1.15, split_pattern=r"[,!?;:।\n]+"):
        if first_audio_ms is None:
            first_audio_ms = (time.perf_counter() - t_tts_start) * 1000
        total_audio_chunks += 1

t_tts_end = time.perf_counter()
tts_latency = (t_tts_end - t_tts_start) * 1000
total_pipeline = (t_tts_end - t_start) * 1000

print(f"Reply text: '{reply_text}'")
print(f"1. STT (Estimated)       : ~200 ms")
print(f"2. LLM ({active_model_name})     : {llm_latency:.0f} ms")
print(f"3. TTS (Kokoro Inproc)   : {tts_latency:.0f} ms (First audio: {first_audio_ms:.0f} ms)")
print(f"TOTAL END-TO-END ROUNDTRIP: {total_pipeline + 200:.0f} ms")
