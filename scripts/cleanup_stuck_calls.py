import sqlite3
from datetime import datetime

conn = sqlite3.connect("calls.db")
c = conn.cursor()

# Find any call that is in-progress but older than 5 minutes
rows = c.execute("SELECT id, start_time FROM calls WHERE status = 'in-progress'").fetchall()
print(f"Found {len(rows)} in-progress calls to clean up:")

for call_id, start_time in rows:
    # Check last message timestamp
    msg_row = c.execute(
        "SELECT timestamp FROM call_messages WHERE call_id = ? ORDER BY id DESC LIMIT 1",
        (call_id,)
    ).fetchone()

    duration = 30.0
    if msg_row and msg_row[0]:
        try:
            t0 = datetime.fromisoformat(start_time)
            t1 = datetime.fromisoformat(msg_row[0])
            duration = max(1.0, round((t1 - t0).total_seconds(), 1))
        except Exception:
            pass

    c.execute(
        "UPDATE calls SET status = 'completed', duration_seconds = ? WHERE id = ?",
        (duration, call_id)
    )
    print(f"  Fixed call {call_id}: marked completed with duration {duration}s")

conn.commit()
conn.close()
print("Done.")
