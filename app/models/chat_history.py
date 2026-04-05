from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from ..core.database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True)
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
