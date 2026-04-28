from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

_MEMORY_STORE: dict[str, InMemoryChatMessageHistory] = {}
MEMORY_WINDOW = 10 # keep only the last human+AI turn pairs in context.


def get_memory(conversation_id: str) -> InMemoryChatMessageHistory:
    if conversation_id not in _MEMORY_STORE:
        _MEMORY_STORE[conversation_id] = InMemoryChatMessageHistory()
    return _MEMORY_STORE[conversation_id]


def save_turn(conversation_id: str, human_message: str, ai_message: str) -> None:
    memory = get_memory(conversation_id)
    memory.add_user_message(human_message)
    memory.add_ai_message(ai_message)
    # Suggested so Trim to window: keep only the last MEMORY_WINDOW*2 messages
    max_messages = MEMORY_WINDOW * 2
    if len(memory.messages) > max_messages:
        memory.messages = memory.messages[-max_messages:]


def get_history_messages(conversation_id: str) -> list[BaseMessage]:
    return get_memory(conversation_id).messages


def clear_memory(conversation_id: str) -> None:
    if conversation_id in _MEMORY_STORE:
        del _MEMORY_STORE[conversation_id]