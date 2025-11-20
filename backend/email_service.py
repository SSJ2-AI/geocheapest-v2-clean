import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        # In production, initialize SendGrid/SES client here
        pass

    async def send_welcome_email(self, email: str, name: str):
        """Send welcome email to new user"""
        logger.info(f"EMAIL: Sending welcome email to {email} ({name})")
        # Mock sending
        print(f"--------------------------------------------------")
        print(f"To: {email}")
        print(f"Subject: Welcome to GeoCheapest!")
        print(f"Body: Hi {name}, welcome to the platform...")
        print(f"--------------------------------------------------")

    async def send_order_confirmation(self, email: str, order_id: str, total: float):
        """Send order confirmation"""
        logger.info(f"EMAIL: Sending order confirmation to {email} for order {order_id}")
        print(f"--------------------------------------------------")
        print(f"To: {email}")
        print(f"Subject: Order Confirmation #{order_id}")
        print(f"Body: Your order of ${total:.2f} has been received.")
        print(f"--------------------------------------------------")

    async def send_password_reset(self, email: str, token: str):
        """Send password reset link"""
        logger.info(f"EMAIL: Sending password reset to {email}")
        print(f"--------------------------------------------------")
        print(f"To: {email}")
        print(f"Subject: Password Reset")
        print(f"Body: Click here to reset: /reset-password?token={token}")
        print(f"--------------------------------------------------")
