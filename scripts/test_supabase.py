"""
Supabase Connection & Table Verification Utility.
Run:
    python scripts/test_supabase.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 printing in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env", override=True)

import backend.database as db


def test_supabase():
    print("==================================================")
    print("  Supabase Database Connection & Schema Verifier ")
    print("==================================================")

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()

    if not url or not key:
        print("[!] SUPABASE_URL or SUPABASE_KEY is missing in your .env file.")
        print("    Add them to .env:")
        print("    SUPABASE_URL=https://your-project.supabase.co")
        print("    SUPABASE_KEY=your-anon-or-service-role-key")
        print("\n[i] Currently running with local SQLite fallback (calls.db).")
        return False

    print(f"[*] Testing connection to: {url[:30]}...")

    client = db.get_supabase_client()
    if not client:
        print("[!] Failed to instantiate Supabase client. Check credentials.")
        return False

    # Attempt auto-migration if missing
    try:
        from backend.database.migrator import auto_migrate_supabase
        auto_migrate_supabase()
    except Exception as e:
        print(f"[!] Auto-migration check: {e}")

    # Check 'calls' table
    try:
        res = client.table("calls").select("id").limit(1).execute()
        print("[OK] Successfully queried table: 'calls'")
    except Exception as e:
        print(f"[!] Error querying table 'calls': {e}")
        return False

    # Check 'call_messages' table
    try:
        res = client.table("call_messages").select("id").limit(1).execute()
        print("[OK] Successfully queried table: 'call_messages'")
    except Exception as e:
        print(f"[!] Error querying table 'call_messages': {e}")
        print("    Did you run backend/database/schema.sql in the Supabase SQL editor?")
        return False

    print("\n[SUCCESS] Supabase PostgreSQL is completely configured and ready for live calls!")
    return True


if __name__ == "__main__":
    test_supabase()
