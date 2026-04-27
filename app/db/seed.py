"""
db/seed.py
──────────
Populates the functional_process table with the 5 processes (A–E).
Run once at startup via init_db() or manually:  python -m app.db.seed
"""

from app.db.database import engine, SessionLocal, Base
from app.db.models import FunctionalProcess, RagDocumentRegistry


PROCESSES = [
    {
        "process_code":            "A",
        "process_name":            "Clarification Handling / Inquiry Resolution",
        "team_area_responsible":   "Customer Service – Tier 1",
        "average_time_till_solved": "2h",
        "atention_channel":        "app, web, whatsapp, phone",
        "priority_level":          "medium",
    },
    {
        "process_code":            "B",
        "process_name":            "Product Cancellation",
        "team_area_responsible":   "Retention & Cancellations Team",
        "average_time_till_solved": "24h",
        "atention_channel":        "app, web, branch",
        "priority_level":          "high",
    },
    {
        "process_code":            "C",
        "process_name":            "Incident Escalation",
        "team_area_responsible":   "Operations – Escalation Desk",
        "average_time_till_solved": "4h",
        "atention_channel":        "phone, web, branch",
        "priority_level":          "critical",
    },
    {
        "process_code":            "D",
        "process_name":            "Customer Data Update",
        "team_area_responsible":   "KYC & Compliance Team",
        "average_time_till_solved": "3 business days",
        "atention_channel":        "branch, app, web",
        "priority_level":          "medium",
    },
    {
        "process_code":            "E",
        "process_name":            "Internal Complaint Management",
        "team_area_responsible":   "Quality & Complaints Unit",
        "average_time_till_solved": "5 business days",
        "atention_channel":        "web, email, branch",
        "priority_level":          "high",
    },
]


def init_db():
    """
    Creates all tables and seeds reference data.
    Safe to call multiple times — skips existing rows.
    """
    print("[seed] Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("[seed] Tables created (or already existed).")
 
    db = SessionLocal()
    try:
        seeded = 0
        for data in PROCESSES:
            exists = db.query(FunctionalProcess).filter_by(
                process_code=data["process_code"]
            ).first()
            if not exists:
                db.add(FunctionalProcess(**data))
                seeded += 1
 
        db.commit()
        if seeded:
            print(f"[seed] functional_process: {seeded} row(s) inserted.")
        else:
            print("[seed] functional_process: already seeded, skipped.")
    except Exception as e:
        db.rollback()
        print(f"[seed] ERROR seeding DB: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
