"""
History Tracking Module for logging user activities.
Tracks data cleaning, predictions, and system events.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from src.logger import setup_logging

logger = setup_logging(__name__)

# History file path
DATA_DIR = "data"
HISTORY_FILE = "history.json"


class HistoryTracker:
    """
    Tracks and logs user activities and system events.
    Stores history in a JSON file for persistence.
    """
    
    # Event types
    EVENT_TYPES = {
        'DATA_UPLOAD': 'Data Upload',
        'DATA_CLEAN': 'Data Cleaning',
        'MODEL_TRAIN': 'Model Training',
        'BATCH_PREDICT': 'Batch Prediction',
        'SINGLE_PREDICT': 'Single Prediction',
        'USER_LOGIN': 'User Login',
        'USER_LOGOUT': 'User Logout',
        'EXPORT_DATA': 'Data Export'
    }
    
    def __init__(self, data_dir: str = DATA_DIR):
        """Initialize the history tracker."""
        self.data_dir = data_dir
        self.history_path = os.path.join(data_dir, HISTORY_FILE)
        self._ensure_history_file()
    
    def _ensure_history_file(self) -> None:
        """Ensure history file exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        if not os.path.exists(self.history_path):
            self._write_history({'events': [], 'metadata': {'created_at': datetime.now().isoformat()}})
    
    def _read_history(self) -> Dict[str, Any]:
        """Read history from file."""
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {'events': []}
    
    def _write_history(self, data: Dict[str, Any]) -> bool:
        """Write history to file."""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error writing history: {e}")
            return False
    
    def log_event(
        self,
        event_type: str,
        description: str,
        user: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log an event to history.
        
        Args:
            event_type: Type of event (use EVENT_TYPES constants).
            description: Human-readable description.
            user: Username who triggered the event.
            details: Additional event details.
            
        Returns:
            True if logged successfully.
        """
        event = {
            'id': self._generate_id(),
            'type': event_type,
            'type_label': self.EVENT_TYPES.get(event_type, event_type),
            'description': description,
            'user': user,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        history = self._read_history()
        history['events'].insert(0, event)  # Most recent first
        
        # Keep only last 500 events
        history['events'] = history['events'][:500]
        
        success = self._write_history(history)
        if success:
            logger.info(f"Logged event: {event_type} - {description}")
        return success
    
    def _generate_id(self) -> str:
        """Generate unique event ID."""
        return datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get filtered events from history.
        
        Args:
            event_type: Filter by event type.
            user: Filter by username.
            limit: Maximum events to return.
            
        Returns:
            List of event dictionaries.
        """
        history = self._read_history()
        events = history.get('events', [])
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.get('type') == event_type]
        
        if user:
            events = [e for e in events if e.get('user') == user]
        
        return events[:limit]
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent activity."""
        return self.get_events(limit=limit)
    
    def get_cleaning_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get data cleaning history."""
        return self.get_events(event_type='DATA_CLEAN', limit=limit)
    
    def get_prediction_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get prediction history."""
        batch = self.get_events(event_type='BATCH_PREDICT', limit=limit)
        single = self.get_events(event_type='SINGLE_PREDICT', limit=limit)
        combined = sorted(batch + single, key=lambda x: x['timestamp'], reverse=True)
        return combined[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get history statistics."""
        history = self._read_history()
        events = history.get('events', [])
        
        stats = {
            'total_events': len(events),
            'by_type': {}
        }
        
        for event in events:
            event_type = event.get('type', 'UNKNOWN')
            stats['by_type'][event_type] = stats['by_type'].get(event_type, 0) + 1
        
        return stats
    
    def clear_history(self) -> bool:
        """Clear all history."""
        return self._write_history({
            'events': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'cleared_at': datetime.now().isoformat()
            }
        })


# Convenience functions for logging specific events
def log_data_upload(filename: str, rows: int, user: Optional[str] = None):
    """Log a data upload event."""
    tracker.log_event(
        'DATA_UPLOAD',
        f"Uploaded {filename} ({rows} rows)",
        user=user,
        details={'filename': filename, 'rows': rows}
    )

def log_data_clean(filename: str, stats: Dict[str, Any], user: Optional[str] = None):
    """Log a data cleaning event."""
    tracker.log_event(
        'DATA_CLEAN',
        f"Cleaned {filename}: {stats.get('initial_rows', 0)} → {stats.get('final_rows', 0)} rows",
        user=user,
        details={'filename': filename, 'stats': stats}
    )

def log_model_train(metrics: Dict[str, Any], user: Optional[str] = None):
    """Log a model training event."""
    tracker.log_event(
        'MODEL_TRAIN',
        f"Model trained: Accuracy {metrics.get('accuracy', 0)*100:.1f}%",
        user=user,
        details={'metrics': metrics}
    )

def log_batch_prediction(filename: str, total: int, high_risk: int, user: Optional[str] = None):
    """Log a batch prediction event."""
    tracker.log_event(
        'BATCH_PREDICT',
        f"Predictions on {filename}: {high_risk}/{total} high risk",
        user=user,
        details={'filename': filename, 'total': total, 'high_risk': high_risk}
    )

def log_user_login(username: str):
    """Log a user login event."""
    tracker.log_event(
        'USER_LOGIN',
        f"User {username} logged in",
        user=username
    )

def log_user_logout(username: str):
    """Log a user logout event."""
    tracker.log_event(
        'USER_LOGOUT',
        f"User {username} logged out",
        user=username
    )


# Singleton instance
tracker = HistoryTracker()
