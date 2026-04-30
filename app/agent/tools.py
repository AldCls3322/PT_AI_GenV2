from sqlalchemy.orm import Session
from app.rag.retriever import retrieve_as_context
from app.db.models import FunctionalProcess

# Get db of processes

def lookup_all_processes(db: Session) -> list[dict]:
    """
    Return all rows from functional_process as a list of dicts.
    Injected into the LLM prompt so it can reason over the full catalogue.
    """
    rows = db.query(FunctionalProcess).order_by(FunctionalProcess.process_code).all()
    return [row.to_dict() for row in rows]

def lookup_process_by_code(process_code: str, db: Session) -> dict | None:
    """
    Fetch a single process row by its code (A–E).
    Used after the LLM identifies which process applies.
    """
    row = db.query(FunctionalProcess).filter_by(process_code=process_code).first()
    return row.to_dict() if row else None

# def lookup_process_db(process_code: str, db: Session) -> dict | None:
#     # process_code : "A"-"E"
#     # db : SQLAlchemy session
#     row = db.query(FunctionalProcess).filter_by(process_code=process_code).first()
#     # Get process metadata from the functional_process table.
#     return row.to_dict() if row else None


# RAG
def rag_search(query: str, process_code: str | None = None) -> tuple[str, list[str]]:
    context, sources, used_ret = retrieve_as_context(query)
    #(context_string, sources_list, rag_used)
    return context, sources, used_ret

# prompt formatting injestion
def format_processes_for_prompt(processes: list[dict]) -> str:
    """
    Format all process rows into a readable block for the system prompt.
    The LLM uses this to identify which process matches the user's question.
    """
    lines = []
    for p in processes:
        lines.append(
            f"  [{p['process_code']}] {p['process_name']}\n"
            f"      Team        : {p['team_area_responsible']}\n"
            f"      Resolution  : {p['average_time_till_solved']}\n"
            f"      Channels    : {p['atention_channel']}\n"
            f"      Priority    : {p['priority_level']}"
        )
    return "\n\n".join(lines)