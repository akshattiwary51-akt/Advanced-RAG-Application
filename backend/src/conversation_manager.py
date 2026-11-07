"""Conversation persistence (SQLite via SQLAlchemy)."""
from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import select

from database.db import Database
from database.models import Conversation, Message, utcnow


class ConversationNotFoundError(Exception):
    pass


class ConversationManager:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _title_from(text: str) -> str:
        t = " ".join(text.split())
        return (t[:60] + "…") if len(t) > 60 else (t or "New conversation")

    def create(self, title: Optional[str] = None) -> dict:
        with self.db.session() as s:
            c = Conversation(title=(title or "New conversation")[:120])
            s.add(c)
            s.flush()
            return c.to_dict()

    def ensure(self, conversation_id: Optional[str], first_message: str) -> str:
        """Return an existing conversation id, or create one titled from the first message."""
        if conversation_id is None:
            with self.db.session() as s:
                c = Conversation(title=self._title_from(first_message))
                s.add(c)
                s.flush()
                return c.id
        with self.db.session() as s:
            if s.get(Conversation, conversation_id) is None:
                raise ConversationNotFoundError(conversation_id)
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str,
                    citations: Optional[list] = None) -> dict:
        with self.db.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None:
                raise ConversationNotFoundError(conversation_id)
            msg = Message(conversation_id=conversation_id, role=role, content=content,
                          citations_json=json.dumps(citations) if citations else None)
            s.add(msg)
            conv.updated_at = utcnow()
            s.flush()
            return msg.to_dict()

    def history(self, conversation_id: str, limit: int = 12) -> list[dict]:
        """Most recent ``limit`` messages as ``{"role", "content"}`` (oldest first)."""
        with self.db.session() as s:
            rows = s.scalars(
                select(Message).where(Message.conversation_id == conversation_id)
                .order_by(Message.id.desc()).limit(limit)
            ).all()
        return [{"role": m.role, "content": m.content} for m in reversed(rows)]

    def get(self, conversation_id: str) -> dict:
        with self.db.session() as s:
            c = s.get(Conversation, conversation_id)
            if c is None:
                raise ConversationNotFoundError(conversation_id)
            return c.to_dict(include_messages=True)

    def list(self, limit: int = 100) -> list[dict]:
        with self.db.session() as s:
            rows = s.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)).all()
            return [c.to_dict() for c in rows]

    def delete(self, conversation_id: str) -> None:
        with self.db.session() as s:
            c = s.get(Conversation, conversation_id)
            if c is None:
                raise ConversationNotFoundError(conversation_id)
            s.delete(c)
