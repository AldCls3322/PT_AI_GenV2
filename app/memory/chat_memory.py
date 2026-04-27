"""
memory/chat_memory.py
─────────────────────
Per-session conversation memory — LangChain 0.3.x compatible.

Uses InMemoryChatMessageHistory (the modern replacement for
ConversationBufferWindowMemory) with a manual window trim so we
keep only the last MEMORY_WINDOW human+AI turn pairs in context.

For production: swap _MEMORY_STORE dict for a Redis-backed store.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# Global registry: conversation_id -> InMemoryChatMessageHistory
_MEMORY_STORE: dict[str, InMemoryChatMessageHistory] = {}

# Number of recent human/AI turn PAIRS to keep in the window
MEMORY_WINDOW = 10


def get_memory(conversation_id: str) -> InMemoryChatMessageHistory:
    """Retrieve or create a message history for the given conversation."""
    if conversation_id not in _MEMORY_STORE:
        _MEMORY_STORE[conversation_id] = InMemoryChatMessageHistory()
    return _MEMORY_STORE[conversation_id]


def save_turn(conversation_id: str, human_message: str, ai_message: str) -> None:
    """
    Append one human→AI turn and trim the window so we never exceed
    MEMORY_WINDOW pairs (2 * MEMORY_WINDOW total messages).
    """
    memory = get_memory(conversation_id)
    memory.add_user_message(human_message)
    memory.add_ai_message(ai_message)

    # Trim to window: keep only the last MEMORY_WINDOW*2 messages
    max_messages = MEMORY_WINDOW * 2
    if len(memory.messages) > max_messages:
        memory.messages = memory.messages[-max_messages:]


def get_history_messages(conversation_id: str) -> list[BaseMessage]:
    """
    Return the windowed list of BaseMessage objects for a conversation.
    Passed directly into the LLM prompt by the orchestrator.
    """
    return get_memory(conversation_id).messages


def clear_memory(conversation_id: str) -> None:
    """Wipe the memory for a conversation (e.g. after session ends)."""
    if conversation_id in _MEMORY_STORE:
        del _MEMORY_STORE[conversation_id]