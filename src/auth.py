"""
Authentication module for user management.
Provides secure password hashing, login, registration, and session management.
"""

import hashlib
import secrets
import re
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import logging
from src.logger import setup_logging
from src.database import db

logger = setup_logging(__name__)


class AuthError(Exception):
    """Custom exception for authentication errors."""
    pass


class AuthService:
    """Service class for authentication operations."""
    
    # Password requirements
    MIN_PASSWORD_LENGTH = 6
    
    def __init__(self):
        """Initialize the auth service."""
        self.db = db
    
    # ==================== PASSWORD UTILITIES ====================
    
    @staticmethod
    def _generate_salt() -> str:
        """Generate a random salt for password hashing."""
        return secrets.token_hex(16)
    
    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """
        Hash a password with the given salt using SHA-256.
        
        Args:
            password: Plain text password.
            salt: Salt for hashing.
            
        Returns:
            Hashed password string.
        """
        combined = f"{salt}{password}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """
        Verify a password against a stored hash.
        
        Args:
            password: Plain text password to verify.
            stored_hash: The stored password hash.
            salt: The salt used for the stored hash.
            
        Returns:
            True if password matches, False otherwise.
        """
        computed_hash = self._hash_password(password, salt)
        return secrets.compare_digest(computed_hash, stored_hash)
    
    # ==================== VALIDATION ====================
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """
        Validate email format.
        
        Args:
            email: Email address to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        if not email:
            return False, "Email is required"
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        return True, ""
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """
        Validate username format.
        
        Args:
            username: Username to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        if not username:
            return False, "Username is required"
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if len(username) > 30:
            return False, "Username must be less than 30 characters"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        
        return True, ""
    
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        if not password:
            return False, "Password is required"
        
        if len(password) < self.MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {self.MIN_PASSWORD_LENGTH} characters"
        
        return True, ""
    
    # ==================== REGISTRATION ====================
    
    def register(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str = "",
        role: str = "user"
    ) -> Tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Desired username.
            email: User's email address.
            password: User's password (will be hashed).
            full_name: User's full name (optional).
            role: User role (default: "user").
            
        Returns:
            Tuple of (success, message).
        """
        # Validate inputs
        valid, error = self.validate_username(username)
        if not valid:
            return False, error
        
        valid, error = self.validate_email(email)
        if not valid:
            return False, error
        
        valid, error = self.validate_password(password)
        if not valid:
            return False, error
        
        # Check if username exists
        if self.db.get_user_by_username(username):
            return False, "Username already exists"
        
        # Check if email exists
        if self.db.get_user_by_email(email):
            return False, "Email already registered"
        
        # Hash password
        salt = self._generate_salt()
        password_hash = self._hash_password(password, salt)
        
        # Create user data
        user_data = {
            "username": username,
            "email": email.lower(),
            "password_hash": password_hash,
            "salt": salt,
            "full_name": full_name,
            "role": role,
            "is_active": True,
            "last_login": None
        }
        
        # Save to database
        success = self.db.create_user(user_data)
        
        if success:
            logger.info(f"User registered successfully: {username}")
            return True, "Registration successful!"
        else:
            logger.error(f"Failed to register user: {username}")
            return False, "Registration failed. Please try again."
    
    # ==================== LOGIN ====================
    
    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticate a user.
        
        Args:
            username: Username or email.
            password: User's password.
            
        Returns:
            Tuple of (success, message, user_data).
        """
        if not username or not password:
            return False, "Username and password are required", None
        
        # Try to find user by username or email
        user = self.db.get_user_by_username(username)
        if not user:
            user = self.db.get_user_by_email(username)
        
        if not user:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return False, "Invalid username or password", None
        
        # Check if user is active
        if not user.get("is_active", True):
            return False, "Account is deactivated", None
        
        # Verify password
        stored_hash = user.get("password_hash", "")
        salt = user.get("salt", "")
        
        if not self._verify_password(password, stored_hash, salt):
            logger.warning(f"Failed login attempt for user: {username}")
            return False, "Invalid username or password", None
        
        # Update last login
        self.db.update_user(user["username"], {"last_login": datetime.now().isoformat()})
        
        # Return sanitized user data (without sensitive info)
        safe_user = {
            "id": user.get("id"),
            "username": user.get("username"),
            "email": user.get("email"),
            "full_name": user.get("full_name"),
            "role": user.get("role"),
            "created_at": user.get("created_at")
        }
        
        logger.info(f"User logged in successfully: {username}")
        return True, "Login successful!", safe_user
    
    # ==================== PASSWORD MANAGEMENT ====================
    
    def change_password(
        self,
        username: str,
        current_password: str,
        new_password: str
    ) -> Tuple[bool, str]:
        """
        Change a user's password.
        
        Args:
            username: Username of the account.
            current_password: Current password for verification.
            new_password: New password to set.
            
        Returns:
            Tuple of (success, message).
        """
        # Verify current password
        success, _, user = self.login(username, current_password)
        if not success:
            return False, "Current password is incorrect"
        
        # Validate new password
        valid, error = self.validate_password(new_password)
        if not valid:
            return False, error
        
        # Hash new password
        salt = self._generate_salt()
        password_hash = self._hash_password(new_password, salt)
        
        # Update user
        success = self.db.update_user(username, {
            "password_hash": password_hash,
            "salt": salt
        })
        
        if success:
            logger.info(f"Password changed for user: {username}")
            return True, "Password changed successfully"
        else:
            return False, "Failed to change password"
    
    # ==================== USER MANAGEMENT ====================
    
    def get_user_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get a user's public profile.
        
        Args:
            username: Username to look up.
            
        Returns:
            User profile dict or None.
        """
        user = self.db.get_user_by_username(username)
        if not user:
            return None
        
        # Return only safe fields
        return {
            "id": user.get("id"),
            "username": user.get("username"),
            "email": user.get("email"),
            "full_name": user.get("full_name"),
            "role": user.get("role"),
            "created_at": user.get("created_at"),
            "last_login": user.get("last_login")
        }
    
    def deactivate_user(self, username: str) -> Tuple[bool, str]:
        """
        Deactivate a user account.
        
        Args:
            username: Username to deactivate.
            
        Returns:
            Tuple of (success, message).
        """
        success = self.db.update_user(username, {"is_active": False})
        if success:
            return True, "Account deactivated"
        return False, "Failed to deactivate account"


# Singleton instance for easy import
auth_service = AuthService()
