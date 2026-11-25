"""Application-level exceptions used by Task 4."""


class Task4Error(Exception):
    """Base exception for expected Task 4 failures."""


class Task2IntegrationError(Task4Error):
    """Raised when the existing Task 2 RAG services cannot be loaded."""


class ChatGenerationError(Task4Error):
    """Raised when Gemini cannot produce a streamed answer."""


class ConversationStoreError(Task4Error):
    """Raised when persistent conversation storage cannot be read or written."""


class ConversationOwnershipError(ConversationStoreError):
    """Raised when a conversation ID is not owned by the supplied browser identity."""
