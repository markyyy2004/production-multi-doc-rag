import os
import uuid
import pytest
from database import init_db, register_user, verify_user, save_chat_message, load_user_chat_history

def test_user_authentication_flow():
    init_db()
    # Generate unique test username to avoid SQLite duplicate collisions
    test_user = f"user_{uuid.uuid4().hex[:8]}"
    test_pass = "password123"

    # 1. Registration
    assert register_user(test_user, test_pass) is True
    # 2. Prevent Duplicates
    assert register_user(test_user, "newpassword") is False
    # 3. Successful Verification
    user = verify_user(test_user, test_pass)
    assert user is not None
    assert user["username"] == test_user
    # 4. Failed Verification
    assert verify_user(test_user, "wrong_pass") is None

def test_chat_persistence_flow():
    init_db()
    test_user = f"user_{uuid.uuid4().hex[:8]}"
    register_user(test_user, "password123")
    user = verify_user(test_user, "password123")
    user_id = user["id"]
    
    save_chat_message(user_id, "user", "What is Java Metaspace?")
    save_chat_message(user_id, "assistant", "Metaspace handles class metadata outside the JVM heap.")
    
    history = load_user_chat_history(user_id)
    assert len(history) >= 2
    assert history[-2]["content"] == "What is Java Metaspace?"
    assert history[-1]["content"] == "Metaspace handles class metadata outside the JVM heap."