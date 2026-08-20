import sqlite3
import hashlib
import json
import os

DB_FILE = "nexus_rag.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def register_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.strip().lower(), hash_password(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username: str, password: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), hash_password(password))
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"id": user[0], "username": user[1]}
    return None

def save_chat_message(user_id: int, role: str, content: str, sources: list = None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    sources_raw = []
    if sources:
        for doc in sources:
            sources_raw.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
    sources_json = json.dumps(sources_raw) if sources_raw else None
    
    cursor.execute(
        "INSERT INTO chat_history (user_id, role, content, sources_json) VALUES (?, ?, ?, ?)",
        (user_id, role, content, sources_json)
    )
    conn.commit()
    conn.close()

def load_user_chat_history(user_id: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, sources_json FROM chat_history WHERE user_id = ? ORDER BY id ASC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for role, content, sources_json in rows:
        msg = {"role": role, "content": content}
        if sources_json:
            msg["sources_json"] = json.loads(sources_json)
        messages.append(msg)
    return messages

def clear_user_chat_history(user_id: int):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()