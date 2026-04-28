
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from app.db.database import Base


class FunctionalProcess(Base):

    __tablename__ = "functional_process"

    process_id            = Column(Integer, primary_key=True, index=True)
    process_code          = Column(String(2), unique=True, nullable=False) # A.B.C.D.E
    process_name          = Column(String(120), nullable=False)
    team_area_responsible = Column(String(100), nullable=False)
    average_time_till_solved = Column(String(40), nullable=False)
    atention_channel      = Column(String(80), nullable=False)
    priority_level        = Column(String(20), nullable=False)

    def to_dict(self) -> dict:
        return {
            "process_id":              self.process_id,
            "process_code":            self.process_code,
            "process_name":            self.process_name,
            "team_area_responsible":   self.team_area_responsible,
            "average_time_till_solved": self.average_time_till_solved,
            "atention_channel":        self.atention_channel,
            "priority_level":          self.priority_level,
        }


class RagDocumentRegistry(Base):
    __tablename__ = "rag_document_registry"
    id              = Column(Integer, primary_key=True, index=True)
    process_code    = Column(String(2), nullable=False)        # "A"–"E"
    document_name   = Column(String(255), nullable=False)
    chunk_count     = Column(Integer, nullable=False)
    embedding_model = Column(String(100), nullable=False)
    chunk_size      = Column(Integer, nullable=False)
    chunk_overlap   = Column(Integer, nullable=False)
    ingested_at     = Column(DateTime, default=datetime.utcnow)
    notes           = Column(Text, nullable=True)
