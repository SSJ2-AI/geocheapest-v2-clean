"""
Firestore database connection for GeoCheapest v2
"""
import logging
import os
from typing import Optional
from google.cloud import firestore

logger = logging.getLogger(__name__)

_db_client: Optional[firestore.Client] = None


def _create_client() -> firestore.Client:
    project_id = os.getenv("GCP_PROJECT_ID")
    return firestore.Client(project=project_id or None)


class FirestoreProxy:
    """Lazy Firestore client that avoids initialization at import time."""

    def _ensure_client(self) -> firestore.Client:
        global _db_client
        if _db_client is None:
            try:
                _db_client = _create_client()
                logger.info("Firestore connected")
            except Exception as exc:
                logger.error(f"Firestore failed: {exc}")
                _db_client = None
        if _db_client is None:
            raise RuntimeError("Firestore client is not initialized")
        return _db_client

    def __getattr__(self, name):
        client = self._ensure_client()
        return getattr(client, name)


db = FirestoreProxy()

# Collection names
STORES = "stores"
PRODUCTS = "products"
SHOPIFY_LISTINGS = "shopifyListings"
AFFILIATE_PRODUCTS = "affiliateProducts"
ORDERS = "orders"
ORDER_ITEMS = "orderItems"
SELLER_PAYOUTS = "sellerPayouts"
COMMISSION_RATES = "commissionRates"
VENDOR_COMMISSION_OVERRIDES = "vendorCommissionOverrides"
RETURN_REQUESTS = "returnRequests"
OAUTH_NONCES = "oauth_nonces"
USERS = "users"
