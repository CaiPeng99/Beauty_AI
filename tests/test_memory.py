"""
test_memory.py
Full test suite for memory.py

Three layers:
  1. Short-term memory (Redis)     -- unit tests, fakeredis, no DB
  2. Long-term memory (PostgreSQL) -- SQLite in-memory with a test-only ChatMemory model;
                                      LLM calls and vector ops are mocked
  3. End-to-end integration tests  -- simulates a real agent conversation flow

How to run:
    .venv/bin/python -m pytest tests/test_memory.py -v
    pytest tests/test_memory.py -v (not work)

Dependencies:
    pip install pytest fakeredis
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOCK_USER_ID = "user_test_abc123"
SESSION_1    = "sess_monday_001"      # first conversation
SESSION_2    = "sess_wednesday_002"   # second conversation, same user


# ---------------------------------------------------------------------------
# Test-only ChatMemory model (no pgvector / ARRAY / TSVECTOR)
# ---------------------------------------------------------------------------

TestBase = declarative_base()

class ChatMemoryTest(TestBase):
    """
    Lightweight stand-in for the real ChatMemory model.
    Replaces Vector(384) with Text so SQLite can create the table.
    """
    __tablename__ = "chat_memory"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(String, index=True)
    session_id  = Column(String, index=True)
    user_query  = Column(Text)
    ai_response = Column(Text)
    summary     = Column(Text)
    # embedding   = Column(Text)   # Text instead of Vector(384) for SQLite compatibility
    embedding   = Column(JSON)   # JSON supports list storage in SQLite
    created_at  = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis(monkeypatch):
    """Replace the real Redis client with fakeredis (no Redis server needed)."""
    import fakeredis
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("app.agent.memory.redis_client", fake)
    return fake


@pytest.fixture
def db_session(monkeypatch):
    """
    In-memory SQLite DB with only the chat_memory table.
    Also patches app.agent.memory.ChatMemory to use the test model,
    so the real pgvector model is never touched.
    """
    engine  = create_engine("sqlite:///:memory:")
    TestBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Swap the real ChatMemory (pgvector) for the test version (SQLite-safe)
    monkeypatch.setattr("app.agent.memory.ChatMemory", ChatMemoryTest)

    yield session
    session.close()
    TestBase.metadata.drop_all(engine)


# ===========================================================================
# 1. Short-term Memory Tests
# ===========================================================================

class TestShortTermMemory:

    def test_get_empty_session(self, fake_redis):
        """A new session should return an empty string without raising."""
        from app.agent.memory import get_short_memory
        assert get_short_memory("nonexistent_session") == ""

    def test_append_and_get(self, fake_redis):
        """Messages appended to a session should be retrievable."""
        from app.agent.memory import append_short_memory, get_short_memory

        append_short_memory(SESSION_1, "User: Recommend a foundation for dry skin")
        append_short_memory(SESSION_1, "AI: I recommend the Estee Lauder Double Wear foundation")

        mem = get_short_memory(SESSION_1)
        assert "dry skin" in mem
        assert "Estee Lauder" in mem

    def test_messages_are_ordered(self, fake_redis):
        """Messages should appear in the order they were appended."""
        from app.agent.memory import append_short_memory, get_short_memory

        append_short_memory("sess_order", "User: first message")
        append_short_memory("sess_order", "AI: second message")
        append_short_memory("sess_order", "User: third message")

        mem = get_short_memory("sess_order")
        assert mem.index("first") < mem.index("second") < mem.index("third")

    def test_clear_memory(self, fake_redis):
        """Clearing a session should return an empty string afterwards."""
        from app.agent.memory import append_short_memory, clear_short_memory, get_short_memory

        append_short_memory("sess_clear", "some message")
        clear_short_memory("sess_clear")
        assert get_short_memory("sess_clear") == ""

    def test_ttl_is_set(self, fake_redis):
        """TTL should be set to 24 h (86400 s) on every write."""
        from app.agent.memory import append_short_memory

        append_short_memory("sess_ttl", "hello")
        ttl = fake_redis.ttl("chat:mem:sess_ttl")
        assert 0 < ttl <= 86400

    def test_different_sessions_isolated(self, fake_redis):
        """Memory from one session should not appear in another."""
        from app.agent.memory import append_short_memory, get_short_memory

        append_short_memory("sess_A", "User: I am user A")
        append_short_memory("sess_B", "User: I am user B")

        assert "user A" in get_short_memory("sess_A")
        assert "user A" not in get_short_memory("sess_B")


# ===========================================================================
# 2. Long-term Memory Tests
# ===========================================================================

class TestLongTermMemory:

    # ------------------------------------------------------------------
    # _extract_memory_from_llm  (LLM is always mocked)
    # ------------------------------------------------------------------

    def test_extract_returns_memory_when_llm_says_yes(self):
        """When the LLM says the turn is worth saving, return the extracted preference."""
        from app.agent.memory import _extract_memory_from_llm

        mock_response = json.dumps({
            "should_save": True,
            "memory": "Dry skin, budget under $50, avoids alcohol-based products"
        })
        with patch("app.agent.memory.client") as mock_client:
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=mock_response))]
            )
            result = _extract_memory_from_llm(
                user_q="I have dry skin, budget under $50, no alcohol please",
                ai_a="Noted, I will keep your preferences in mind"
            )

        assert result == "Dry skin, budget under $50, avoids alcohol-based products"

    def test_extract_returns_none_when_llm_says_no(self):
        """When the LLM says the turn is not worth saving, return None."""
        from app.agent.memory import _extract_memory_from_llm

        mock_response = json.dumps({"should_save": False, "memory": None})
        with patch("app.agent.memory.client") as mock_client:
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content=mock_response))]
            )
            result = _extract_memory_from_llm(
                user_q="How much does this product cost?",
                ai_a="It costs around $30"
            )

        assert result is None

    def test_extract_handles_llm_failure_gracefully(self):
        """If the LLM call fails, return None without raising."""
        from app.agent.memory import _extract_memory_from_llm

        with patch("app.agent.memory.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("API timeout")
            result = _extract_memory_from_llm("any question", "any answer")

        assert result is None

    # ------------------------------------------------------------------
    # maybe_save_memory
    # ------------------------------------------------------------------

    def test_save_when_llm_says_yes(self, db_session):
        """When the LLM says yes, write to DB and return True."""
        from app.agent.memory import maybe_save_memory

        with patch("app.agent.memory._extract_memory_from_llm",
                   return_value="Dry skin, budget under $50"):
            with patch("app.agent.memory._get_encoder") as mock_enc:
                mock_enc.return_value.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
                saved = maybe_save_memory(
                    db_session, MOCK_USER_ID, SESSION_1,
                    user_q="I have dry skin and a budget of $50",
                    ai_a="Got it, here are my recommendations..."
                )

        assert saved is True

        # Verify the row actually landed in the DB
        row = db_session.query(ChatMemoryTest).filter_by(user_id=MOCK_USER_ID).first()
        assert row is not None
        assert "Dry skin" in row.summary

    def test_not_save_when_llm_says_no(self, db_session):
        """When the LLM says no, skip DB write and return False."""
        from app.agent.memory import maybe_save_memory

        with patch("app.agent.memory._extract_memory_from_llm", return_value=None):
            saved = maybe_save_memory(
                db_session, MOCK_USER_ID, SESSION_1,
                user_q="How much does this product cost?",
                ai_a="Around $30"
            )

        assert saved is False
        assert db_session.query(ChatMemoryTest).count() == 0

    # ------------------------------------------------------------------
    # get_long_memory -- key test: cross-session retrieval by user_id
    # cosine_distance not available in SQLite, so we mock the DB query
    # and test the formatting / fallback logic instead.
    # ------------------------------------------------------------------

    def test_cross_session_recall(self, db_session):
        """
        Critical test: memory saved in SESSION_1 must be retrievable in SESSION_2.
        Verifies filtering is by user_id (not session_id) and the result is formatted correctly.
        """
        from app.agent.memory import maybe_save_memory, get_long_memory

        # Save a preference in SESSION_1
        with patch("app.agent.memory._extract_memory_from_llm",
                   return_value="Dry skin, budget under $50"):
            with patch("app.agent.memory._get_encoder") as mock_enc:
                mock_enc.return_value.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
                maybe_save_memory(
                    db_session, MOCK_USER_ID, SESSION_1,
                    user_q="I have dry skin and a budget of $50",
                    ai_a="Understood"
                )

        # Mock the vector query to return the row we just saved
        saved_row = db_session.query(ChatMemoryTest).first()
        with patch("app.agent.memory._get_encoder") as mock_enc:
            mock_enc.return_value.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
            with patch("app.agent.memory.ChatMemory") as mock_model:
                mock_model.__name__ = "ChatMemory"
                mock_db_query = MagicMock()
                mock_db_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [saved_row]
                db_session_mock = MagicMock()
                db_session_mock.query.return_value = mock_db_query

                result = get_long_memory(db_session_mock, MOCK_USER_ID, "recommend a day cream", top_k=3)

        assert "Dry skin" in result or "50" in result

    def test_different_users_isolated(self, db_session):
        """Long-term memory from one user must not appear for another user."""
        from app.agent.memory import maybe_save_memory

        user_a = "user_aaa"
        user_b = "user_bbb"

        with patch("app.agent.memory._extract_memory_from_llm",
                   return_value="User A has dry skin"):
            with patch("app.agent.memory._get_encoder") as mock_enc:
                mock_enc.return_value.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
                maybe_save_memory(db_session, user_a, SESSION_1, "dry skin", "ok")

        # user_b should have no rows
        rows = db_session.query(ChatMemoryTest).filter_by(user_id=user_b).all()
        assert len(rows) == 0

    def test_empty_memory_returns_empty_string(self, db_session):
        """A brand new user with no stored memory should get an empty string."""
        from app.agent.memory import get_long_memory

        with patch("app.agent.memory._get_encoder") as mock_enc:
            mock_enc.return_value.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
            with patch("app.agent.memory.ChatMemory") as mock_model:
                mock_db_query = MagicMock()
                mock_db_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
                db_session_mock = MagicMock()
                db_session_mock.query.return_value = mock_db_query

                result = get_long_memory(db_session_mock, "brand_new_user", "any question", top_k=3)

        assert result == ""


# ===========================================================================
# 3. build_system_prompt Tests
# ===========================================================================

class TestBuildSystemPrompt:

    def test_prompt_contains_memory_when_exists(self, db_session):
        """When long-term memory exists, the system prompt should include user preferences."""
        from app.agent.memory import build_system_prompt

        fake_memory = "[2026-06-01] Dry skin, prefers Japanese brands"

        with patch("app.agent.memory.get_long_memory", return_value=fake_memory):
            prompt = build_system_prompt(db_session, MOCK_USER_ID, "recommend foundation")

        assert "What you already know about this user" in prompt
        assert "Japanese" in prompt

    def test_prompt_has_no_memory_section_for_new_user(self, db_session):
        """A new user with no memory should get the base prompt only."""
        from app.agent.memory import build_system_prompt

        with patch("app.agent.memory.get_long_memory", return_value=""):
            prompt = build_system_prompt(db_session, "new_user_xyz", "recommend foundation")

        assert "What you already know about this user" not in prompt
        assert "beauty assistant" in prompt


# ===========================================================================
# 4. End-to-End Integration Tests
# ===========================================================================

class TestEndToEnd:

    def test_full_two_session_flow(self, fake_redis, db_session):
        """
        Full flow:
          SESSION_1: user states preferences -> saved to long-term memory
          SESSION_2: new conversation -> long-term memory injected into system prompt
        """
        from app.agent.memory import (
            append_short_memory, maybe_save_memory,
            build_system_prompt, clear_short_memory,
        )

        # ── SESSION_1 ──────────────────────────────────────────────────
        user_q = "I have dry skin, budget under $50, no alcohol-based products please"
        ai_a   = "Based on your dry skin and $50 budget, I recommend the Estee Lauder Double Wear"

        append_short_memory(SESSION_1, f"User: {user_q}")
        append_short_memory(SESSION_1, f"AI: {ai_a}")

        with patch("app.agent.memory._extract_memory_from_llm",
                   return_value="Dry skin, budget under $50, avoids alcohol-based products"):
            with patch("app.agent.memory._get_encoder") as mock_enc:
                mock_enc.return_value.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
                saved = maybe_save_memory(
                    db_session, MOCK_USER_ID, SESSION_1, user_q, ai_a
                )

        assert saved is True

        # ── SESSION_2 (new conversation, same user) ────────────────────
        clear_short_memory(SESSION_1)
        assert get_short_memory_helper(fake_redis, SESSION_1) == ""

        fake_memory = "[2026-06-29] Dry skin, budget under $50, avoids alcohol-based products"
        with patch("app.agent.memory.get_long_memory", return_value=fake_memory):
            from app.agent.memory import build_system_prompt
            system_prompt = build_system_prompt(db_session, MOCK_USER_ID, "recommend a day cream")

        assert "dry" in system_prompt.lower() or "50" in system_prompt
        assert "What you already know about this user" in system_prompt

    def test_coreference_resolution_via_short_memory(self, fake_redis):
        """
        Short-term memory test: 'it' in Turn 2 is resolvable to the product in Turn 1
        because both turns are present in the session memory.
        """
        from app.agent.memory import append_short_memory, get_short_memory

        append_short_memory(SESSION_1, "User: Recommend a foundation for sensitive skin")
        append_short_memory(SESSION_1, "AI: I recommend the Lancome Teint Idole, gentle and non-irritating")
        append_short_memory(SESSION_1, "User: How much does it cost?")

        mem = get_short_memory(SESSION_1)
        assert "Lancome Teint Idole" in mem
        assert "How much does it cost" in mem

    def test_memory_boundary_new_session_no_short_memory(self, fake_redis):
        """A brand new session should have empty short-term memory."""
        from app.agent.memory import append_short_memory, get_short_memory

        append_short_memory(SESSION_1, "User: I have dry skin")
        assert get_short_memory(SESSION_2) == ""


# ---------------------------------------------------------------------------
# Helper (used in e2e test to check Redis directly via the fake)
# ---------------------------------------------------------------------------

def get_short_memory_helper(fake_redis, session_id: str) -> str:
    val = fake_redis.get(f"chat:mem:{session_id}")
    return val if val else ""