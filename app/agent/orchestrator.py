"""
Main Banorte Orchestrator Agent

Must
Accept a user message and conversation_id.
Classify the intent into one of 5 processesA,B,C,D,E.
Get process metadata from the db.
Get relevant knowledge from ChromaDB (RAG).
Create prompting,include:
    - System instructions
    - Process context (from DB)
    - RAG knowledge excerpts
    - Conversation history (from memory)
Call the LLM and return a structured response.
Persist the turn to memory.
"""

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.config import settings
# todo from app.memory.chat_memory import get_memory, save_turn, get_history_messages
# todo from app.agent.tools import classify_process, lookup_process_db, rag_search
from app.schemas.request_schema import ChatRequest, ChatResponse, ProcessInfo


#! MUST FROM REQUIRMENTS: PROCESS_LABELS
PROCESS_LABELS = {
    "A": "Clarification Handling / Inquiry Resolution",
    "B": "Product Cancellation",
    "C": "Incident Escalation",
    "D": "Customer Data Update",
    "E": "Internal Complaint Management",
}

SYSTEM_PROMPT_TEMPLATE = """
You are a professional, empathetic virtual assistant for Banorte, one of Mexico's banks. Your role is to help clients resolve their banking needs efficiently.

Guidelines:
- Always respond in the same language the client uses (Spanish or English).
- Be concise, clear, and respectful.
- Never invent bank policies — only use the information provided.
- If you cannot resolve an issue, escalate gracefully.
- For security-sensitive topics (passwords, full card numbers), never ask for them.

Current detected process: {process_code} — {process_name}
Responsible team: {team_area_responsible}
Expected resolution time: {average_time_till_solved}
Available channels: {atention_channel}
Priority: {priority_level}

Relevant knowledge from the Banorte knowledge base:
───────────────────────────────────────────────────
{rag_context}
───────────────────────────────────────────────────

Use the above information to answer the client's question accurately.
If the knowledge base has no relevant information, rely on general banking best practices.
""".strip()


def _build_llm():
    """Instantiate the LLM based on LLM_PROVIDER setting."""
    if settings.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.3,
        )
    # Default: OpenAI
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.3,
        max_tokens=800,
    )


def run_agent(request: ChatRequest, db: Session) -> ChatResponse:
    """
    Main orchestration function by API route.

    1.  Classify user intent → process_code
    2.  Fetch DB metadata for that process
    3.  Run RAG retrieval
    4.  Build prompt (system + history + user message)
    5.  Call LLM
    6.  Save turn to memory
    7.  Return ChatResponse

    request : Pydantic ChatRequest
    db : SQLAlchemy session
    """
    user_text       = request.message.text
    conversation_id = request.conversation_id
    user_id         = request.user_id

    process_code = classify_process(user_text)
    print(f"[Orchestrator] 🔍  Detected process: {process_code} — {PROCESS_LABELS[process_code]}")

    process_data = lookup_process_db(process_code, db)
    if not process_data:
        # Graceful degradation — use minimal defaults
        process_data = {
            "process_id":              0,
            "process_code":            process_code,
            "process_name":            PROCESS_LABELS[process_code],
            "team_area_responsible":   "Customer Service",
            "average_time_till_solved": "N/A",
            "atention_channel":        "app, web, phone",
            "priority_level":          "medium",
        }

    # Rag usage call 
    rag_context, sources = rag_search(user_text, process_code)
    print(f"[Orchestrator] 📚  RAG sources: {sources or ['none']}")
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        process_code             = process_code,
        process_name             = process_data["process_name"],
        team_area_responsible    = process_data["team_area_responsible"],
        average_time_till_solved = process_data["average_time_till_solved"],
        atention_channel         = process_data["atention_channel"],
        priority_level           = process_data["priority_level"],
        rag_context              = rag_context,
    )
    messages = [SystemMessage(content=system_content)]

    #conversation history
    history = get_history_messages(conversation_id)
    messages.extend(history)
    # Append the current user turn
    messages.append(HumanMessage(content=user_text))

    # LLM
    llm    = _build_llm()
    result = llm.invoke(messages)
    reply  = result.content.strip()
    print(f"[Orchestrator] 💬  Reply generated ({len(reply)} chars)")
    # memory usage
    save_turn(conversation_id, user_text, reply)

    # get response format
    process_info = ProcessInfo(
        process_id               = process_data["process_id"],
        process_name             = process_data["process_name"],
        team_area_responsible    = process_data["team_area_responsible"],
        average_time_till_solved = process_data["average_time_till_solved"],
        atention_channel         = process_data["atention_channel"],
        priority_level           = process_data["priority_level"],
    )

    return ChatResponse(
        conversation_id  = conversation_id,
        user_id          = user_id,
        reply            = reply,
        detected_process = f"{process_code} — {PROCESS_LABELS[process_code]}",
        process_info     = process_info,
        sources          = sources,
    )
