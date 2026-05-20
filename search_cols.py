import sqlite3

db_path = r"C:\Users\Guhankarthik\AppData\Local\label-studio\label-studio\label_studio.sqlite3"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    
    for t in tables:
        cursor.execute(f"PRAGMA table_info({t})")
        cols = [c[1] for c in cursor.fetchall()]
        for c in cols:
            if "legacy" in c.lower() or "token" in c.lower() or "auth" in c.lower():
                print(f"Table: {t}, Column: {c}")
except Exception as e:
    print(f"Error: {e}")
