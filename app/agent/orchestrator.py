import re
from langchain.schema import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session
from app.config import settings

from app.memory.chat_memory import save_turn, get_history_messages
from app.agent.tools import lookup_all_processes, lookup_process_by_code, rag_search, format_processes_for_prompt
from app.schemas.request_schema import ChatRequest, ChatResponse, ProcessInfo


# MUST FROM REQUIRMENTS: PROCESS_LABELS
PROCESS_LABELS = {
    "A": "Atencion de aclaraciones",
    "B": "Cancelacion de productos",
    "C": "Escalamiento de incidencias",
    "D": "Actualizacion de datos del cliente",
    "E": "Gestion de quejas internas",
}

SYSTEM_PROMPT_TEMPLATE = """
Eres un asistente virtual profesional y empatico para Banorte, uno de los bancos de Mexico. Tu funcion es ayudar a los clientes a resolver sus necesidades bancarias de manera eficiente.

Reglamento de Idioma Critico:
- Detecta el idioma (español o inglés) del usuario.
- Responde SIEMPRE en el mismo idioma, sin importar el idioma de la base de conocimientos o del proceso detectado.

Guidelines:
- Siempre responda en el idioma que usa el cliente (español o inglés).
- Sea conciso, claro y respetuoso.
- Basa las respuestas en la informacion proporcionada, evita inventar documentacion de politicas bancarias. Si se inventan politicas bancarias describe explicitamente que fue generado por IA y no forma parte de las politicas oficiales de Banorte.
- Si no puede resolver un problema, escalelo de forma profesional.
- Nunca solicite información confidencial (contraseñas, numeros completos de tarjeta).
- Referencia conversaciones previas si son relevantes, pero no asuma que el cliente las recuerda.

════════════════════════════════════════════════════════
BANORTE PROCESS CATALOGUE (from internal database)
════════════════════════════════════════════════════════
{process_catalogue}
════════════════════════════════════════════════════════
 
{rag_section}

════════════════════════════════════════════════════════
TU OBJETIVO:
════════════════════════════════════════════════════════
1. Leer el mensaje del ususario cuidadosamente.
2. Identifica que procesp del catalogo/process_catalogue mejor empata con su pregunta.
3. {synthesis_instruction}
4. Al final completo de tu respuesta, en una nueva linea, desplega el proceso detectado
   codigo usando este formato exacto — sin espacios, sin texto extra:
   PROCESS_CODE:[X]
   Where X is A, B, C, D, or E. Example: PROCESS_CODE:[B]
""".strip()

RAG_FOUND_SECTION = """
════════════════════════════════════════════════════════
KNOWLEDGE BASE EXCERPTS (from document store)
════════════════════════════════════════════════════════
{rag_context}
════════════════════════════════════════════════════════"""
 
RAG_NOT_FOUND_SECTION = """
════════════════════════════════════════════════════════
KNOWLEDGE BASE: No relevant documents found for this query.
Use only the process catalogue above to answer.
════════════════════════════════════════════════════════"""
 
SYNTHESIS_WITH_RAG = (
    "Build your answer by combining BOTH sources: "
    "use the Knowledge Base Excerpts for detailed procedural information, "
    "AND use the Process Catalogue for operational data (team, channels, "
    "resolution time, priority). Your answer must reflect BOTH — do not "
    "omit the channels, team, or resolution time from the catalogue."
)
 
SYNTHESIS_DB_ONLY = (
    "Answer using only the Process Catalogue above. "
    "Include the responsible team, all attention channels, resolution time, "
    "and priority level in your answer."
)

def _build_llm():
    if settings.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2,
        )
    # Default: OpenAI
    # reducir temperatura para evitar inventar respuestas.
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2,
        max_tokens=1800,
    )

# parser
def _extract_process_code(llm_reply: str) -> str | None:
    """
    Parse the PROCESS_CODE:[X] tag the LLM appends to its reply.
    Returns the code letter (A-E) or None if the tag is missing/malformed.
    """
    match = re.search(r"PROCESS_CODE:\[([A-E])\]", llm_reply, re.IGNORECASE)
    return match.group(1).upper() if match else None
 
 
def _strip_process_tag(text: str) -> str:
    """Remove the PROCESS_CODE:[X] line from the visible reply."""
    return re.sub(r"\n?PROCESS_CODE:\[[A-E]\]\s*$", "", text, flags=re.IGNORECASE).strip()

# main
def run_agent(request: ChatRequest, db: Session) -> ChatResponse:
    # FLOW
    # 1 RAG search
    # 2 DB fetch (procesos)
    # 3 Build prompt — DB + RAG (or not-found notice)
    # 4 Inject memory — add conversation history
    # 5 LLM call — invoke, reply + PROCESS_CODE tag
    # 6 Parse tag — parse detected process code from reply
    # 7 DB enrich — fetch full process row for the detected code
    # 8 Save memory
    user_text = request.message.text
    conversation_id = request.conversation_id
    user_id = request.user_id

    # RAG
    rag_context, sources, rag_used = rag_search(user_text)
    print(f"RAG used: {rag_used} | sources: {sources or ['none']}")

    # DB
    all_processes    = lookup_all_processes(db)
    process_catalogue = format_processes_for_prompt(all_processes)
    print(f"DB catalogue loaded: {len(all_processes)} processes")

    if rag_used:
        rag_section          = RAG_FOUND_SECTION.format(rag_context=rag_context)
        synthesis_instruction = SYNTHESIS_WITH_RAG
    else:
        rag_section          = RAG_NOT_FOUND_SECTION
        synthesis_instruction = SYNTHESIS_DB_ONLY
 
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        process_catalogue     = process_catalogue,
        rag_section           = rag_section,
        synthesis_instruction = synthesis_instruction,
    )

    # 4 inject memory
    messages   = [SystemMessage(content=system_content)]
    history    = get_history_messages(conversation_id)
    prior_turns = len(history) // 2
    messages.extend(history)
    messages.append(HumanMessage(content=user_text))
    print(f"Prompt: 1 system + {prior_turns} history turn(s) + 1 user = {len(messages)} messages")
    

    #? debug prompt message
    # print("REQUEST:")
    # print(f"{messages}")

    # LLM OpenAI
    llm        = _build_llm()
    result     = llm.invoke(messages)
    raw_reply  = result.content.strip()
 
    # ── Step 6: Parse detected process from LLM output ───────────────────────
    detected_code = _extract_process_code(raw_reply)
    clean_reply   = _strip_process_tag(raw_reply)
    print(f"Detected process tag: {detected_code or 'not found — defaulting to A'}")
    print(f"Reply generated ({len(clean_reply)} chars)")

    if detected_code:
        process_data = lookup_process_by_code(detected_code, db)
    else:
        print(f"Answer could not be found to fit any process.")
        detected_code = "X"
        process_data  = None

    if not process_data:
        #! Edge-case scenario
        process_data = {
            "process_id":              0,
            "process_code":            detected_code,
            "process_name":            PROCESS_LABELS.get(detected_code, "Unknown Inquiry"),
            "team_area_responsible":   "Customer Service",
            "average_time_till_solved": "N/A",
            "atention_channel":        "app, web, phone",
            "priority_level":          "low",
        }
    
    # 8
    save_turn(conversation_id, user_text, clean_reply)

    process_info = ProcessInfo(
        process_id               = process_data["process_id"],
        process_name             = process_data["process_name"],
        team_area_responsible    = process_data["team_area_responsible"],
        average_time_till_solved = process_data["average_time_till_solved"],
        atention_channel         = process_data["atention_channel"],
        priority_level           = process_data["priority_level"],
    )
 
    detected_label = f"{detected_code} — {PROCESS_LABELS.get(detected_code, 'General Inquiry')}"
 
    return ChatResponse(
        conversation_id   = conversation_id,
        user_id           = user_id,
        reply             = clean_reply,
        detected_process  = detected_label,
        process_info      = process_info,
        sources           = sources,
        memory_turns_used = prior_turns,
    )


