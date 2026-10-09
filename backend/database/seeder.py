"""
Supabase Database Connection Verifier & Automatic Seeder for IndiaWalls AI Voice Calling.
Exclusively operates on Supabase Cloud PostgreSQL. Local SQLite has been completely removed.
"""

from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    import backend.config
    from backend.database.connection import get_supabase_client, is_supabase_enabled
    from backend.database.repository import check_supabase_schema, init_db
else:
    import backend.config
    from .connection import get_supabase_client, is_supabase_enabled
    from .repository import check_supabase_schema, init_db

logger = logging.getLogger("hindi-ai-calling.db.seeder")

SEED_CALLS: List[Dict[str, Any]] = [
    {
        "id": "seed_call_stone_wall",
        "call_sid": "seed_sid_99101",
        "channel": "exotel",
        "from_number": "09876543210",
        "to_number": "08047104999",
        "duration_seconds": 145.0,
        "turn_count": 6,
        "status": "completed",
        "time_offset_mins": 35,
        "messages": [
            ("user", "नमस्ते, मुझे स्टोन डिज़ाइन बाउंड्री वॉल के बारे में जानना था।"),
            (
                "assistant",
                "जी सर, नमस्कार! इंडियावाल्स इन्फ्राटेक में आपका स्वागत है। हमारी स्टोन डिज़ाइन वॉल बहुत ही प्रीमियम कंक्रीट वॉल है, जो बिल्कुल प्राकृतिक पत्थर जैसा लुक देती है।",
            ),
            ("user", "इसकी न्यूनतम ऊंचाई और रेट क्या रहता है?"),
            (
                "assistant",
                "सर, इसकी न्यूनतम ऊंचाई 6 फीट से शुरू होती है और आवश्यकतानुसार 10 या 12 फीट तक लगाई जा सकती है। यह पारंपरिक ईंट की दीवार से 20 से 30 प्रतिशत सस्ती पड़ती है।",
            ),
            ("user", "क्या मुझे इसका कोटेशन और साइट विजिट मिल सकती है?"),
            (
                "assistant",
                "जी बिल्कुल सर! आप हमारे नंबर 7820879777 पर कॉल कर सकते हैं या हमें indiawallsofficial@gmail.com पर मेल कर सकते हैं। हमारी टीम तुरंत कोटेशन भेज देगी।",
            ),
        ],
    },
    {
        "id": "seed_call_farmhouse_wall",
        "call_sid": "seed_sid_99102",
        "channel": "exotel",
        "from_number": "09123456789",
        "to_number": "08047104999",
        "duration_seconds": 98.0,
        "turn_count": 4,
        "status": "completed",
        "time_offset_mins": 15,
        "messages": [
            ("user", "क्या आप फार्महाउस के लिए बाउंड्री वॉल का काम करते हैं?"),
            (
                "assistant",
                "जी सर! हमारे पास फार्महाउस, एग्रीकल्चरल और रेजिडेंशियल प्रोजेक्ट्स के लिए 6 प्रकार की शानदार प्रीकास्ट कंक्रीट दीवारें उपलब्ध हैं।",
            ),
            ("user", "इंस्टॉलेशन में कितना समय लग जाता है?"),
            (
                "assistant",
                "सर, प्रीकास्ट तकनीक की वजह से यह बहुत ही तेज़ है। कुछ ही दिनों में पूरी बाउंड्री बनकर तैयार हो जाती है, बिना प्लास्टर या पेंट के झंझट के।",
            ),
        ],
    },
]


def seed_supabase_database(force: bool = False) -> int:
    """Check Supabase database and seed initial demo data if empty."""
    if not is_supabase_enabled():
        logger.warning("[DB Seeder] Supabase credentials not found in .env.")
        return 0

    if not check_supabase_schema():
        logger.info("[DB Seeder] Supabase tables not ready yet. Skipping cloud seeding.")
        return 0

    client = get_supabase_client()
    if not client:
        return 0

    now = datetime.now()
    seeded_count = 0

    try:
        # Check current row count in Supabase
        res = client.table("calls").select("id").limit(5).execute()
        count = len(res.data or [])
        if count > 0 and not force:
            logger.info(f"[DB Seeder] Supabase 'calls' table already has records ({count}+). No seed needed.")
            return 0

        logger.info("[DB Seeder] Supabase cloud is empty. Auto-seeding initial demo calls & transcripts...")

        for call in SEED_CALLS:
            call_start = (now - timedelta(minutes=call["time_offset_mins"])).isoformat()
            call_end = (now - timedelta(minutes=call["time_offset_mins"] - 2)).isoformat()

            client.table("calls").upsert({
                "id": call["id"],
                "call_sid": call["call_sid"],
                "channel": call["channel"],
                "from_number": call["from_number"],
                "to_number": call["to_number"],
                "start_time": call_start,
                "end_time": call_end,
                "duration_seconds": call["duration_seconds"],
                "turn_count": call["turn_count"],
                "status": call["status"],
            }).execute()

            msg_dt = datetime.fromisoformat(call_start)
            for idx, (role, content) in enumerate(call["messages"]):
                msg_time = (msg_dt + timedelta(seconds=idx * 15)).isoformat()
                client.table("call_messages").insert({
                    "call_id": call["id"],
                    "role": role,
                    "content": content,
                    "timestamp": msg_time,
                    "latency_ms": 450 if role == "assistant" else 0,
                }).execute()

            seeded_count += 1

        logger.info(f"[DB Seeder] Successfully seeded {seeded_count} demo call(s) into Supabase Cloud.")
    except Exception as e:
        logger.warning(f"[DB Seeder] Supabase seeding error: {e}")

    return seeded_count


def verify_and_seed_database(force: bool = False) -> Dict[str, Any]:
    """
    Main startup verifier for Supabase:
    1. Tests Supabase connection.
    2. Checks schema tables ('calls', 'call_messages').
    3. Auto-seeds initial demo calls if empty.
    """
    logger.info("[DB Check] Verifying Supabase Cloud connection and seed status...")

    if not is_supabase_enabled():
        return {
            "supabase_status": "Missing credentials in .env",
            "supabase_calls_count": 0,
            "supabase_newly_seeded": 0,
        }

    try:
        from .migrator import auto_migrate_supabase
        auto_migrate_supabase()
    except Exception as e:
        logger.debug(f"Auto-migration attempt notice: {e}")

    schema_ready = check_supabase_schema(force_refresh=True)
    newly_seeded = 0
    total_calls = 0

    if schema_ready:
        newly_seeded = seed_supabase_database(force=force)
        client = get_supabase_client()
        if client:
            try:
                res = client.table("calls").select("id").execute()
                total_calls = len(res.data or [])
            except Exception:
                pass

    status_str = "Connected & Active" if schema_ready else "Pending schema.sql execution"

    logger.info(
        f"[DB Check] Supabase Status: {status_str} | Total Calls in Cloud: {total_calls} | "
        f"Newly Seeded: {newly_seeded}"
    )

    return {
        "supabase_status": status_str,
        "supabase_calls_count": total_calls,
        "supabase_newly_seeded": newly_seeded,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    force_flag = "--force" in sys.argv
    res = verify_and_seed_database(force=force_flag)
    print("\n=== Supabase Database Status Report ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
