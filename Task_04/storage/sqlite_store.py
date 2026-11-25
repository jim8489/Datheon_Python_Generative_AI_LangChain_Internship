"""SQLite persistence for browser-isolated conversations and source references."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from core.exceptions import ConversationOwnershipError, ConversationStoreError

logger = logging.getLogger(__name__)
_BROWSER_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class SQLiteConversationStore:
    """Persist chat history scoped to a signed anonymous browser identity.

    Each operation uses a short-lived connection. WAL and a busy timeout help
    separate Streamlit sessions share the same local SQLite database. Calls
    that supply ``previous_state`` use incremental, baseline-aware persistence
    so stale tabs cannot replace messages written by another tab.
    """

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.database_path.parent.chmod(0o700)
        except OSError:
            pass
        self._initialize_schema()
        try:
            self.database_path.chmod(0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=15,
            isolation_level="DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS browser_profiles (
            browser_id TEXT PRIMARY KEY,
            active_conversation_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            browser_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (browser_id) REFERENCES browser_profiles(browser_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_browser_updated
            ON conversations(browser_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(conversation_id, ordinal),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation_ordinal
            ON messages(conversation_id, ordinal);
        """
        try:
            with self._connection() as connection:
                connection.executescript(schema)
                connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                f"Could not initialize the conversation database at {self.database_path}."
            ) from exc

    @staticmethod
    def _validate_browser_id(browser_id: str) -> None:
        if not isinstance(browser_id, str) or not _BROWSER_ID_PATTERN.fullmatch(browser_id):
            raise ValueError("A valid 32-character browser identity is required.")

    @staticmethod
    def _ensure_profile(connection: sqlite3.Connection, browser_id: str, now: str) -> None:
        connection.execute(
            """INSERT OR IGNORE INTO browser_profiles
               (browser_id, active_conversation_id, created_at, updated_at)
               VALUES (?, NULL, ?, ?)""",
            (browser_id, now, now),
        )

    @staticmethod
    def _message_ids(conversation: Any) -> set[str]:
        if not isinstance(conversation, Mapping):
            return set()
        messages = conversation.get("messages", [])
        if not isinstance(messages, list):
            return set()
        return {
            str(message["id"])
            for message in messages
            if isinstance(message, Mapping) and message.get("id")
        }

    @staticmethod
    def _serialize_sources(message: Mapping[str, Any]) -> str:
        sources = message.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        try:
            return json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            logger.warning("Could not serialize source metadata; saving message without sources.")
            return "[]"

    def load_browser_state(self, browser_id: str) -> dict[str, Any]:
        """Load only conversations owned by the supplied browser identity."""
        self._validate_browser_id(browser_id)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._connection() as connection:
                self._ensure_profile(connection, browser_id, now)
                conversation_rows = connection.execute(
                    """SELECT id, title, created_at, updated_at
                       FROM conversations WHERE browser_id = ?
                       ORDER BY updated_at DESC, created_at DESC""",
                    (browser_id,),
                ).fetchall()

                conversations: dict[str, dict[str, Any]] = {}
                for row in conversation_rows:
                    conversations[row["id"]] = {
                        "id": row["id"],
                        "title": row["title"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "messages": [],
                    }

                if conversations:
                    placeholders = ",".join("?" for _ in conversations)
                    message_rows = connection.execute(
                        f"""SELECT id, conversation_id, role, content,
                                   sources_json, created_at
                            FROM messages
                            WHERE conversation_id IN ({placeholders})
                            ORDER BY conversation_id, ordinal""",
                        tuple(conversations),
                    ).fetchall()
                    for message in message_rows:
                        try:
                            sources = json.loads(message["sources_json"] or "[]")
                            if not isinstance(sources, list):
                                sources = []
                        except (TypeError, json.JSONDecodeError):
                            logger.warning(
                                "Ignoring malformed source metadata for message %s",
                                message["id"],
                            )
                            sources = []

                        conversations[message["conversation_id"]]["messages"].append(
                            {
                                "id": message["id"],
                                "role": message["role"],
                                "content": message["content"],
                                "sources": sources,
                                "created_at": message["created_at"],
                            }
                        )

                profile = connection.execute(
                    "SELECT active_conversation_id FROM browser_profiles WHERE browser_id = ?",
                    (browser_id,),
                ).fetchone()
                active_id = profile["active_conversation_id"] if profile else None
                if active_id not in conversations:
                    active_id = next(iter(conversations), None)

                return {
                    "conversations": conversations,
                    "active_conversation_id": active_id,
                }
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not load this browser's conversation history."
            ) from exc

    def save_browser_state(
        self,
        browser_id: str,
        state: Mapping[str, Any],
        previous_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist browser state.

        Without ``previous_state`` this retains the original full-snapshot
        behavior for compatibility with existing callers. Streamlit sessions
        should pass the state snapshot that was last saved by that tab. The
        incremental mode writes only additions and deletions relative to that
        baseline; it never re-inserts baseline messages missing from SQLite.
        """
        self._validate_browser_id(browser_id)
        conversations = state.get("conversations")
        if not isinstance(conversations, Mapping) or not conversations:
            raise ConversationStoreError("At least one conversation is required for persistence.")

        if previous_state is None:
            self._save_full_snapshot(browser_id, state, conversations)
        else:
            self._save_incremental(browser_id, state, conversations, previous_state)

    def _save_full_snapshot(
        self,
        browser_id: str,
        state: Mapping[str, Any],
        conversations: Mapping[str, Any],
    ) -> None:
        """Compatibility path: replace all stored conversations with snapshot."""
        active_id = state.get("active_conversation_id")
        if active_id not in conversations:
            active_id = max(
                conversations,
                key=lambda key: conversations[key].get("updated_at", ""),
            )

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._ensure_profile(connection, browser_id, now)

                for key, conversation in conversations.items():
                    if not isinstance(conversation, Mapping):
                        raise ConversationStoreError("Conversation data must be a mapping.")
                    conversation_id = str(conversation.get("id", key))
                    if conversation_id != str(key):
                        raise ConversationStoreError("Conversation map key does not match its ID.")

                    cursor = connection.execute(
                        """INSERT INTO conversations
                           (id, browser_id, title, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET
                               title = excluded.title,
                               updated_at = excluded.updated_at
                           WHERE conversations.browser_id = excluded.browser_id""",
                        (
                            conversation_id,
                            browser_id,
                            str(conversation.get("title", "New conversation")),
                            str(conversation.get("created_at", now)),
                            str(conversation.get("updated_at", now)),
                        ),
                    )
                    if cursor.rowcount == 0:
                        raise ConversationOwnershipError(
                            "This conversation does not belong to the current browser."
                        )

                    connection.execute(
                        "DELETE FROM messages WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                    messages = conversation.get("messages", [])
                    if not isinstance(messages, list):
                        messages = []
                    for ordinal, message in enumerate(messages):
                        if not isinstance(message, Mapping):
                            continue
                        role = message.get("role")
                        content = message.get("content")
                        if role not in {"user", "assistant"} or not isinstance(content, str):
                            continue
                        connection.execute(
                            """INSERT INTO messages
                               (id, conversation_id, ordinal, role, content, sources_json, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                str(message.get("id") or uuid4().hex),
                                conversation_id,
                                ordinal,
                                role,
                                content,
                                self._serialize_sources(message),
                                str(message.get("created_at", now)),
                            ),
                        )

                retained_ids = {str(key) for key in conversations}
                existing_rows = connection.execute(
                    "SELECT id FROM conversations WHERE browser_id = ?",
                    (browser_id,),
                ).fetchall()
                for row in existing_rows:
                    if row["id"] not in retained_ids:
                        connection.execute(
                            "DELETE FROM conversations WHERE id = ? AND browser_id = ?",
                            (row["id"], browser_id),
                        )

                connection.execute(
                    """UPDATE browser_profiles
                       SET active_conversation_id = ?, updated_at = ?
                       WHERE browser_id = ?""",
                    (active_id, now, browser_id),
                )
        except ConversationStoreError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ConversationStoreError("Conversation data failed database validation.") from exc
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not save this browser's conversation history."
            ) from exc

    def _save_incremental(
        self,
        browser_id: str,
        state: Mapping[str, Any],
        conversations: Mapping[str, Any],
        previous_state: Mapping[str, Any],
    ) -> None:
        """Merge this tab's changes without replacing data from other tabs.

        The baseline represents only what this Streamlit tab last saved or
        loaded. New records created by other tabs are therefore not treated as
        deletions when this tab submits an older in-memory conversation list.
        A baseline conversation/message that has already disappeared from the
        database is not resurrected by a stale tab.
        """
        previous_conversations = previous_state.get("conversations", {})
        if not isinstance(previous_conversations, Mapping):
            previous_conversations = {}

        current_ids = {str(key) for key in conversations}
        previous_ids = {str(key) for key in previous_conversations}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            with self._connection() as connection:
                # Serializes writers before the delta is calculated/applied.
                connection.execute("BEGIN IMMEDIATE")
                self._ensure_profile(connection, browser_id, now)

                # Delete only conversations this tab knew about and explicitly removed.
                for conversation_id in previous_ids - current_ids:
                    connection.execute(
                        "DELETE FROM conversations WHERE id = ? AND browser_id = ?",
                        (conversation_id, browser_id),
                    )

                for key, conversation in conversations.items():
                    if not isinstance(conversation, Mapping):
                        raise ConversationStoreError("Conversation data must be a mapping.")
                    conversation_id = str(conversation.get("id", key))
                    if conversation_id != str(key):
                        raise ConversationStoreError("Conversation map key does not match its ID.")

                    existing = connection.execute(
                        "SELECT browser_id, updated_at FROM conversations WHERE id = ?",
                        (conversation_id,),
                    ).fetchone()

                    if existing is not None and existing["browser_id"] != browser_id:
                        raise ConversationOwnershipError(
                            "This conversation does not belong to the current browser."
                        )

                    if existing is None:
                        # An ID in the baseline was deleted by another tab/user
                        # action. Do not recreate it from this stale snapshot.
                        if conversation_id in previous_ids:
                            continue
                        connection.execute(
                            """INSERT INTO conversations
                               (id, browser_id, title, created_at, updated_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                conversation_id,
                                browser_id,
                                str(conversation.get("title", "New conversation")),
                                str(conversation.get("created_at", now)),
                                str(conversation.get("updated_at", now)),
                            ),
                        )
                    else:
                        incoming_updated = str(conversation.get("updated_at", now))
                        # Do not let an older tab move the activity timestamp back.
                        # Equal-second timestamps are accepted so a title update
                        # made in the same clock second is not lost.
                        connection.execute(
                            """UPDATE conversations
                               SET title = CASE WHEN ? >= updated_at THEN ? ELSE title END,
                                   updated_at = CASE WHEN ? > updated_at THEN ? ELSE updated_at END
                               WHERE id = ? AND browser_id = ?""",
                            (
                                incoming_updated,
                                str(conversation.get("title", "New conversation")),
                                incoming_updated,
                                incoming_updated,
                                conversation_id,
                                browser_id,
                            ),
                        )

                    previous_conversation = previous_conversations.get(conversation_id)
                    if previous_conversation is None:
                        previous_conversation = previous_conversations.get(key)
                    baseline_message_ids = self._message_ids(previous_conversation)

                    messages = conversation.get("messages", [])
                    if not isinstance(messages, list):
                        messages = []
                    current_message_ids = {
                        str(message.get("id"))
                        for message in messages
                        if isinstance(message, Mapping) and message.get("id")
                    }

                    # Clear operations delete only messages in this tab's baseline.
                    for message_id in baseline_message_ids - current_message_ids:
                        connection.execute(
                            "DELETE FROM messages WHERE id = ? AND conversation_id = ?",
                            (message_id, conversation_id),
                        )

                    # Existing baseline IDs are never reinserted. This prevents
                    # stale tabs from resurrecting messages cleared elsewhere.
                    next_ordinal_row = connection.execute(
                        "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal "
                        "FROM messages WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    next_ordinal = int(next_ordinal_row["next_ordinal"])

                    for message in messages:
                        if not isinstance(message, Mapping):
                            continue
                        message_id = str(message.get("id") or uuid4().hex)
                        if message_id in baseline_message_ids:
                            continue
                        role = message.get("role")
                        content = message.get("content")
                        if role not in {"user", "assistant"} or not isinstance(content, str):
                            continue

                        cursor = connection.execute(
                            """INSERT OR IGNORE INTO messages
                               (id, conversation_id, ordinal, role, content, sources_json, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                message_id,
                                conversation_id,
                                next_ordinal,
                                role,
                                content,
                                self._serialize_sources(message),
                                str(message.get("created_at", now)),
                            ),
                        )
                        if cursor.rowcount:
                            next_ordinal += 1

                active_id = state.get("active_conversation_id")
                active_row = None
                if active_id in current_ids:
                    active_row = connection.execute(
                        "SELECT id FROM conversations WHERE id = ? AND browser_id = ?",
                        (str(active_id), browser_id),
                    ).fetchone()

                if active_row is None:
                    active_row = connection.execute(
                        """SELECT id FROM conversations WHERE browser_id = ?
                           ORDER BY updated_at DESC, created_at DESC LIMIT 1""",
                        (browser_id,),
                    ).fetchone()
                persisted_active_id = active_row["id"] if active_row else None

                connection.execute(
                    """UPDATE browser_profiles
                       SET active_conversation_id = ?, updated_at = ?
                       WHERE browser_id = ?""",
                    (persisted_active_id, now, browser_id),
                )
        except ConversationStoreError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ConversationStoreError("Conversation data failed database validation.") from exc
        except sqlite3.Error as exc:
            raise ConversationStoreError(
                "Could not save this browser's conversation history."
            ) from exc
