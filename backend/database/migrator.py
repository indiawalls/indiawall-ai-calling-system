"""
Supabase Schema Auto-Migrator.
Automatically detects and provisions missing database tables ('calls', 'call_messages')
via Direct PostgreSQL connection (psycopg2) or Supabase Management API.
"""

import logging
import os
from pathlib import Path
import re
from typing import Optional
import urllib.request
import json

from backend.config import SUPABASE_KEY, SUPABASE_URL
from backend.database.repository import check_supabase_schema

logger = logging.getLogger("hindi-ai-calling.db.migrator")

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.sql"


def extract_project_ref(url: str) -> Optional[str]:
    """Extract Supabase project reference from SUPABASE_URL."""
    if not url:
        return None
    match = re.search(r"https?://([a-zA-Z0-9_-]+)\.supabase\.(?:co|in)", url)
    return match.group(1) if match else None


def get_schema_sql() -> str:
    """Read the SQL schema contents."""
    if SCHEMA_FILE.exists():
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def migrate_via_postgres(project_ref: str, db_password: str) -> bool:
    """Execute schema.sql via direct PostgreSQL connection (psycopg2)."""
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 is not installed. Run: pip install psycopg2-binary")
        return False

    sql = get_schema_sql()
    if not sql:
        logger.error("schema.sql file not found.")
        return False

    # Connection targets to try: direct and poolers
    connection_strings = [
        f"postgresql://postgres:{db_password}@db.{project_ref}.supabase.co:5432/postgres?sslmode=require",
        f"postgresql://postgres.{project_ref}:{db_password}@aws-0-ap-south-1.pooler.supabase.com:6543/postgres?sslmode=require",
        f"postgresql://postgres.{project_ref}:{db_password}@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require",
        f"postgresql://postgres.{project_ref}:{db_password}@aws-0-us-east-1.pooler.supabase.com:6543/postgres?sslmode=require",
    ]

    for conn_str in connection_strings:
        try:
            logger.info("Connecting to Supabase PostgreSQL to execute schema migration...")
            with psycopg2.connect(conn_str, connect_timeout=8) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql)
            logger.info("✅ Successfully created tables 'calls' and 'call_messages' in Supabase via PostgreSQL!")
            return True
        except Exception as e:
            logger.debug(f"Postgres connection attempt failed: {e}")

    logger.warning("Could not connect to Supabase PostgreSQL using provided SUPABASE_DB_PASSWORD.")
    return False


def migrate_via_management_api(project_ref: str, access_token: str) -> bool:
    """Execute schema.sql via Supabase Management API using personal access token."""
    sql = get_schema_sql()
    if not sql:
        return False

    api_url = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"
    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "Content-Type": "application/json",
    }
    body = json.dumps({"query": sql}).encode("utf-8")

    try:
        req = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status in (200, 201):
                logger.info("✅ Successfully created tables in Supabase via Management API!")
                return True
    except Exception as e:
        logger.warning(f"Supabase Management API migration error: {e}")

    return False


def auto_migrate_supabase() -> bool:
    """
    Check if Supabase schema is ready. If tables are missing, automatically
    attempt to provision them using DB password, Management Token, or DATABASE_URL.
    """
    # 1. Check if already ready
    if check_supabase_schema(force_refresh=True):
        return True

    project_ref = extract_project_ref(SUPABASE_URL)
    if not project_ref:
        logger.error("Could not parse project reference from SUPABASE_URL.")
        return False

    logger.info(f"Supabase tables missing for project '{project_ref}'. Attempting automatic table creation...")

    # 2. Try Method 1: DB Password or DATABASE_URL
    db_password = os.environ.get("SUPABASE_DB_PASSWORD", "").strip()
    database_url = os.environ.get("DATABASE_URL", "").strip()

    if database_url:
        try:
            import psycopg2
            sql = get_schema_sql()
            with psycopg2.connect(database_url, connect_timeout=8) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql)
            logger.info("✅ Successfully created Supabase tables via DATABASE_URL!")
            return check_supabase_schema(force_refresh=True)
        except Exception as e:
            logger.warning(f"DATABASE_URL migration failed: {e}")

    if db_password:
        if migrate_via_postgres(project_ref, db_password):
            return check_supabase_schema(force_refresh=True)

    # 3. Try Method 2: Management Access Token (sbp_...)
    access_token = os.environ.get("SUPABASE_ACCESS_TOKEN", "").strip()
    if access_token:
        if migrate_via_management_api(project_ref, access_token):
            return check_supabase_schema(force_refresh=True)

    # 4. If neither credential is provided, log clear instructions
    sql_url = f"https://supabase.com/dashboard/project/{project_ref}/sql/new"
    logger.warning(
        "\n" + "=" * 75 + "\n"
        "[!] AUTOMATIC TABLE CREATION NOTICE:\n"
        "    The publishable API key cannot run 'CREATE TABLE' statements over HTTP.\n"
        "    To let the code create tables automatically, add ONE of these to your .env:\n"
        "      • SUPABASE_DB_PASSWORD=your_database_password\n"
        "      • SUPABASE_ACCESS_TOKEN=your_token (from https://supabase.com/dashboard/account/tokens)\n"
        "\n"
        f"    OR simply open your SQL editor and run 'backend/database/schema.sql':\n"
        f"    👉 {sql_url}\n"
        + "=" * 75 + "\n"
    )

    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    auto_migrate_supabase()
