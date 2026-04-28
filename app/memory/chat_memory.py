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

    turn_count = len(memory.messages) // 2
    print(f"[Memory] conversation_id='{conversation_id}' | turns stored: {turn_count}/{MEMORY_WINDOW}")



def get_history_messages(conversation_id: str) -> list[BaseMessage]:
    messages = get_memory(conversation_id).messages
    turn_count = len(messages) // 2
 
    if turn_count == 0:
        print(f"[Memory] conversation_id='{conversation_id}' | no prior history (new session)")
    else:
        print(f"[Memory] conversation_id='{conversation_id}' | injecting {turn_count} prior turn(s) into prompt")

    return get_memory(conversation_id).messages

def get_turn_count(conversation_id: str) -> int:
    """Return how many complete turns are stored for a conversation."""
    return len(get_memory(conversation_id).messages) // 2

def clear_memory(conversation_id: str) -> None:
    if conversation_id in _MEMORY_STORE:
        del _MEMORY_STORE[conversation_id]
        print(f"[Memory] conversation_id='{conversation_id}' | memory cleared")