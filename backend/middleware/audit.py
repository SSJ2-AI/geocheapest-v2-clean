import logging
import json
from datetime import datetime
from fastapi import Request
from google.cloud import firestore
from starlette.middleware.base import BaseHTTPMiddleware
from analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, db_client_getter):
        super().__init__(app)
        self.db_client_getter = db_client_getter
        self.analytics = AnalyticsService()

    async def dispatch(self, request: Request, call_next):
        # Only log write methods to sensitive endpoints
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            path = request.url.path
            if any(p in path for p in ["/admin/", "/vendor/", "/auth/"]):
                await self._log_action(request)
        
        response = await call_next(request)
        return response

    async def _log_action(self, request: Request):
        try:
            # Extract user info if available (from auth middleware)
            # Note: This runs before auth in the stack, so we might not have user yet
            # In a real app, we'd use a context var or check auth header manually
            user_id = "anonymous"
            auth_header = request.headers.get("Authorization")
            if auth_header:
                user_id = "authenticated_user" # Placeholder for decoded token

            log_entry = {
                "timestamp": datetime.utcnow(),
                "method": request.method,
                "path": request.url.path,
                "ip_address": request.client.host,
                "user_agent": request.headers.get("user-agent"),
                "user_id": user_id
            }
            
            # We need to get the DB client. In a middleware this is tricky.
            # For now, we'll just log to stdout which is captured by Cloud Logging
            # In production, we'd fire-and-forget to a Pub/Sub topic
            logger.info(f"AUDIT: {json.dumps(log_entry, default=str)}")
            
            # Stream to BigQuery via AnalyticsService
            await self.analytics.log_event("platform_logs", "audit_events", log_entry)
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
