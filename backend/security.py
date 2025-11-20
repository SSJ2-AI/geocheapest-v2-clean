import os
from functools import lru_cache
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken


class TokenCipher:
    """Encrypt/decrypt short-lived secrets (Shopify access tokens)."""

    def __init__(self, key_env: str = "SHOPIFY_TOKEN_ENCRYPTION_KEY"):
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(
                f"Missing encryption key: set {key_env} to a base64 Fernet key"
            )
        self._cipher = Fernet(key.encode())

    def encrypt(self, plaintext: str) -> str:
        """Return URL-safe encrypted token."""
        if plaintext is None:
            raise ValueError("Cannot encrypt empty token")
        token = self._cipher.encrypt(plaintext.encode("utf-8"))
        return token.decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt token or raise if tampered/invalid."""
        if not ciphertext:
            raise ValueError("Cannot decrypt empty token")
        try:
            value = self._cipher.decrypt(ciphertext.encode("utf-8"))
            return value.decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Invalid encrypted token") from exc


@lru_cache(maxsize=1)
def get_token_cipher() -> TokenCipher:
    """Cached accessor so we don't re-create the Fernet object repeatedly."""
    return TokenCipher()


# ==================== AUTHENTICATION ====================
from datetime import datetime, timedelta
from typing import Optional, Union, Any
from passlib.context import CryptContext
from jose import jwt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"sub": str(subject), "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
