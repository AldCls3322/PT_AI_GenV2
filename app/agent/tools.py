from sqlalchemy.orm import Session
from app.rag.retriever import retrieve_as_context
from app.db.models import FunctionalProcess

# Keyword to enhance prompt engineering
_PROCESS_KEYWORDS: list[tuple[str, list[str]]] = [
    ("C", [
        "fraud", "fraude", "stolen", "robado", "hack", "compromised",
        "incident", "incidente", "escalate", "urgent", "urgente", "emergency",
        "block my card", "bloquear tarjeta",
    ]),
    ("B", [
        "cancel", "cancellation", "cancelar", "cancelación", "close account",
        "cerrar cuenta", "give up", "dar de baja", "unsubscribe",
    ]),
    ("E", [
        "complaint", "complain", "queja", "reclamación", "reclamacion",
        "dissatisfied", "insatisfecho", "bad service", "mal servicio",
    ]),
    ("D", [
        "update", "change", "actualizar", "cambiar", "address", "domicilio",
        "phone number", "teléfono", "email", "rfc", "curp", "personal data",
        "datos personales",
    ]),
    ("A", [
        # Catch-all — inquiry / clarification
        "cómo", "qué", "cuándo", "dónde", "por qué", "puedo", "necesito", "help", "ayuda",
        "how", "what", "when", "where", "why", "can i", 
        "information", "información", "question", "pregunta",
    ]),
]


def classify_process(text: str) -> str:
    #! todo: replace with an LLM zero-shot classification call.
    lower = text.lower()
    for code, keywords in _PROCESS_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return code
    return "A"


# DB Lookup
def lookup_process_db(process_code: str, db: Session) -> dict | None:
    # process_code : "A"-"E"
    # db : SQLAlchemy session
    row = db.query(FunctionalProcess).filter_by(process_code=process_code).first()
    # Get process metadata from the functional_process table.
    return row.to_dict() if row else None


# RAG
def rag_search(query: str, process_code: str | None = None) -> tuple[str, list[str]]:
    context, sources = retrieve_as_context(query, process_code)
    #(context_string, sources_list)
    return context, sources
