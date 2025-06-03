from typing import Optional, Dict, Any
from sqlitedict import SqliteDict
import json
import os
from datetime import datetime

class MemoryHandler:
    """
    Handles both short-term (session) and long-term (persistent) memory for the AI pipeline.
    Uses SQLite for persistent storage and in-memory dict for session storage.
    """
    def __init__(self, db_path: str = "memory.sqlite"):
        """
        Initialize memory handler with both session and persistent storage.
        
        Args:
            db_path (str): Path to SQLite database file for persistent storage
        """
        self.session_memory = {}
        self.db_path = db_path
        self.db = SqliteDict(db_path, autocommit=True)
        
    def save_session(self, key: str, data: Any) -> None:
        """Save data to session memory."""
        self.session_memory[key] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_session(self, key: str) -> Optional[Any]:
        """Retrieve data from session memory."""
        return self.session_memory.get(key, {}).get('data')
        
    def save_persistent(self, key: str, data: Any) -> None:
        """Save data to persistent storage."""
        self.db[key] = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.db.commit()
        
    def get_persistent(self, key: str) -> Optional[Any]:
        """Retrieve data from persistent storage."""
        entry = self.db.get(key)
        return entry.get('data') if entry else None
        
    def list_recent_sessions(self, n: int = 5) -> Dict[str, Any]:
        """Get n most recent session entries."""
        sorted_items = sorted(
            self.session_memory.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        return dict(sorted_items[:n])
        
    def clear_session(self) -> None:
        """Clear session memory."""
        self.session_memory.clear()
        
    def __del__(self):
        """Ensure database is properly closed."""
        if hasattr(self, 'db'):
            self.db.close()
