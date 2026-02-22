"""
Database module for JSON-based storage.
Provides CRUD operations for user management.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from src.logger import setup_logging

logger = setup_logging(__name__)

# Default paths
DEFAULT_DATA_DIR = "data"
USERS_FILE = "users.json"


class JSONDatabase:
    """Simple JSON file-based database for user storage."""
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        """
        Initialize the database.
        
        Args:
            data_dir: Directory where JSON files will be stored.
        """
        self.data_dir = data_dir
        self.users_path = os.path.join(data_dir, USERS_FILE)
        self._ensure_data_dir()
        self._ensure_users_file()
    
    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def _ensure_users_file(self) -> None:
        """Ensure the users JSON file exists with default admin user."""
        if not os.path.exists(self.users_path):
            # Create default admin user with pre-hashed password
            # Password: admin123 (hashed with salt below)
            import hashlib
            salt = "default_admin_salt_2026"
            password_hash = hashlib.sha256(f"{salt}admin123".encode()).hexdigest()
            
            default_data = {
                "users": [
                    {
                        "id": 1,
                        "username": "admin",
                        "email": "admin@auca.edu",
                        "password_hash": password_hash,
                        "salt": salt,
                        "full_name": "System Administrator",
                        "role": "admin",
                        "is_active": True,
                        "last_login": None,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                ],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            self._write_json(self.users_path, default_data)
            logger.info(f"Created users file with default admin: {self.users_path}")
    
    def _read_json(self, filepath: str) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error reading {filepath}: {e}")
            return {}
    
    def _write_json(self, filepath: str, data: Dict[str, Any]) -> bool:
        """Write data to a JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error writing to {filepath}: {e}")
            return False
    
    # ==================== USER OPERATIONS ====================
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users from the database."""
        data = self._read_json(self.users_path)
        return data.get("users", [])
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Find a user by username.
        
        Args:
            username: The username to search for.
            
        Returns:
            User dict if found, None otherwise.
        """
        users = self.get_all_users()
        for user in users:
            if user.get("username", "").lower() == username.lower():
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Find a user by email.
        
        Args:
            email: The email to search for.
            
        Returns:
            User dict if found, None otherwise.
        """
        users = self.get_all_users()
        for user in users:
            if user.get("email", "").lower() == email.lower():
                return user
        return None
    
    def create_user(self, user_data: Dict[str, Any]) -> bool:
        """
        Create a new user in the database.
        
        Args:
            user_data: Dictionary containing user information.
            
        Returns:
            True if successful, False otherwise.
        """
        data = self._read_json(self.users_path)
        users = data.get("users", [])
        
        # Generate user ID
        max_id = max([u.get("id", 0) for u in users], default=0)
        user_data["id"] = max_id + 1
        user_data["created_at"] = datetime.now().isoformat()
        user_data["updated_at"] = datetime.now().isoformat()
        
        users.append(user_data)
        data["users"] = users
        
        success = self._write_json(self.users_path, data)
        if success:
            logger.info(f"Created user: {user_data.get('username')}")
        return success
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing user.
        
        Args:
            username: Username of the user to update.
            updates: Dictionary of fields to update.
            
        Returns:
            True if successful, False otherwise.
        """
        data = self._read_json(self.users_path)
        users = data.get("users", [])
        
        for i, user in enumerate(users):
            if user.get("username", "").lower() == username.lower():
                users[i].update(updates)
                users[i]["updated_at"] = datetime.now().isoformat()
                data["users"] = users
                success = self._write_json(self.users_path, data)
                if success:
                    logger.info(f"Updated user: {username}")
                return success
        
        logger.warning(f"User not found for update: {username}")
        return False
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user from the database.
        
        Args:
            username: Username of the user to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        data = self._read_json(self.users_path)
        users = data.get("users", [])
        
        original_count = len(users)
        users = [u for u in users if u.get("username", "").lower() != username.lower()]
        
        if len(users) < original_count:
            data["users"] = users
            success = self._write_json(self.users_path, data)
            if success:
                logger.info(f"Deleted user: {username}")
            return success
        
        logger.warning(f"User not found for deletion: {username}")
        return False


# Singleton instance for easy import
db = JSONDatabase()
