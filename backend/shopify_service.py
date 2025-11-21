"""
Shopify integration service for OAuth, GraphQL bulk sync, and webhooks.
"""
import asyncio
import gzip
import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from google.cloud import firestore

from database import PRODUCTS, SHOPIFY_LISTINGS, STORES, db
from security import TokenCipher, get_token_cipher
from search_service import SearchService
from agent_service import AgentService

logger = logging.getLogger(__name__)
BULK_PRODUCTS_QUERY = """
{
  products {
    edges {
      node {
        id
        title
        productType
        tags
        status
        vendor
        handle
        totalInventory
        createdAt
        updatedAt
        images(first: 10) {
          edges {
            node {
              url
              transformedSrc
              originalSrc
            }
          }
        }
        variants(first: 250) {
          edges {
            node {
              id
              title
              sku
              barcode
              price
              inventoryQuantity
              availableForSale
              requiresShipping
              inventoryItem {
                id
              }
            }
          }
        }
      }
    }
  }
}
"""

WEBHOOK_TOPICS = ["products/update", "inventory_levels/update"]


class ShopifyService:
    """Handle Shopify API interactions (OAuth, product sync, webhooks)."""

    def __init__(self):
        self.search_service = SearchService()
        self.agent_service = AgentService()
        self.api_version = os.getenv("SHOPIFY_API_VERSION", "2024-01")
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.token_cipher = get_token_cipher()

    # ------------------------------------------------------------------
    # OAuth helpers
    # ------------------------------------------------------------------
    def encrypt_token(self, token: str) -> str:
        return self.token_cipher.encrypt(token)

    def decrypt_token(self, token: str) -> str:
        return self.token_cipher.decrypt(token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def sync_products(self, shop: str):
        """Run a Shopify GraphQL bulk sync for the given shop."""
        store_doc = db.collection(STORES).document(shop).get()
        if not store_doc.exists:
            return 0

        store_data = store_doc.to_dict()
        try:
            access_token = self._get_store_token(store_doc, store_data)
        except ValueError:
            return 0

        bulk_id = await self._run_products_bulk_query(shop, access_token)
        if not bulk_id:
            return 0

        bulk_result = await self._wait_for_bulk_operation(shop, access_token)
        if not bulk_result or not bulk_result.get("url"):
            return 0

        processed = 0
        async for record in self._stream_bulk_file(bulk_result["url"]):
            if record.get("__typename") != "Product":
                continue

            normalized = self._normalize_from_graphql(record, store_data)
            if not normalized:
                continue

            product_id = self._upsert_product(normalized)
            self._upsert_listings(store_data, shop, product_id, normalized)
            processed += 1

        store_doc.reference.update(
            {
                "last_sync_at": datetime.utcnow(),
                "total_products": processed,
                "sync_status": "completed",
            }
        )
        return processed

    async def sync_single_product(self, shop: str, product_data: Dict[str, Any]):
        """
        Sync a single product from webhook data
        """
        # ... existing logic ...
        # After updating Firestore, index to Search Engine
        # Note: We need to construct the unified product data here
        
        # AI Normalization: Clean up the data before indexing
        normalized = await self.agent_service.normalize_product(
            product_data.get("title", ""),
            product_data.get("body_html", "")
        )
        
        # For now, we'll just index the raw data as a placeholder
        await self.search_service.upsert_product({
            "id": str(product_data.get("id")),
            "title": normalized.get("canonical_name", product_data.get("title")),
            "shop": shop,
            "updated_at": datetime.utcnow().isoformat(),
            "ai_tags": normalized.get("tags", [])
        })
        """Handle product update webhook payload."""
        store_doc = db.collection(STORES).document(shop).get()
        if not store_doc.exists:
            return

        store_data = store_doc.to_dict()
        normalized = self._normalize_from_rest(product_data, store_data)
        if not normalized:
            return

        product_id = self._upsert_product(normalized)
        self._upsert_listings(store_data, shop, product_id, normalized)

    async def delete_product(self, shop: str, product_id: str):
        """Mark Shopify listings as deleted when a product is removed."""
        listings = (
            db.collection(SHOPIFY_LISTINGS)
            .where("store_id", "==", shop)
            .where("shopify_product_id", "==", str(product_id))
            .stream()
        )
        for listing in listings:
            listing.reference.update(
                {"status": "deleted", "updated_at": datetime.utcnow()}
            )

    async def handle_inventory_level_update(
        self, shop: str, payload: Dict[str, Any]
    ):
        """Update variant quantities from inventory webhook."""
        inventory_item_id = payload.get("inventory_item_id")
        if not inventory_item_id:
            return

        listing_docs = (
            db.collection(SHOPIFY_LISTINGS)
            .where("store_id", "==", shop)
            .where("inventory_item_id", "==", str(inventory_item_id))
            .stream()
        )
        for doc in listing_docs:
            doc.reference.update(
                {"quantity": payload.get("available", 0), "updated_at": datetime.utcnow()}
            )

    async def get_shop_details(self, shop: str, access_token: str) -> Dict[str, Any]:
        """Fetch metadata about the Shopify store."""
        endpoint = f"https://{shop}/admin/api/{self.api_version}/shop.json"
        headers = self._rest_headers(access_token)
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(endpoint, headers=headers)
            if response.status_code == 200:
                return response.json().get("shop", {})
        return {}

    async def ensure_webhooks(self, shop: str, access_token: str):
        """Create or update webhook subscriptions for the vendor."""
        endpoint = f"https://{shop}/admin/api/{self.api_version}/webhooks.json"
        headers = self._rest_headers(access_token)
        callback_url = f"{self.backend_url}/api/shopify/webhook"

        async with httpx.AsyncClient(timeout=20) as client:
            existing_resp = await client.get(endpoint, headers=headers)
            existing = existing_resp.json().get("webhooks", []) if existing_resp.is_success else []
            existing_map = {hook["topic"]: hook for hook in existing}

            for topic in WEBHOOK_TOPICS:
                payload = {
                    "webhook": {
                        "topic": topic,
                        "address": callback_url,
                        "format": "json",
                    }
                }
                hook = existing_map.get(topic)
                if hook and hook.get("address") == callback_url:
                    continue
                if hook:
                    update_url = f"https://{shop}/admin/api/{self.api_version}/webhooks/{hook['id']}.json"
                    await client.put(update_url, headers=headers, json=payload)
                else:
                    await client.post(endpoint, headers=headers, json=payload)

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------
    def _get_store_token(self, store_doc, store_data) -> str:
        encrypted = store_data.get("access_token_encrypted")
        if encrypted:
            return self.decrypt_token(encrypted)

        legacy = store_data.get("access_token")
        if not legacy:
            raise ValueError("Shopify access token missing for store")

        encrypted = self.encrypt_token(legacy)
        store_doc.reference.update(
            {
                "access_token_encrypted": encrypted,
                "access_token_migrated_at": datetime.utcnow(),
            }
        )
        return legacy

    # ------------------------------------------------------------------
    # Bulk operation helpers
    # ------------------------------------------------------------------
    async def _run_products_bulk_query(self, shop: str, token: str) -> Optional[str]:
        mutation = """
        mutation RunBulk($query: String!) {
          bulkOperationRunQuery(query: $query) {
            bulkOperation { id status }
            userErrors { field message }
          }
        }
        """
        payload = {"query": mutation, "variables": {"query": BULK_PRODUCTS_QUERY}}
        response = await self._graphql_request(shop, token, payload)
        data = response.get("data", {}).get("bulkOperationRunQuery", {})
        errors = data.get("userErrors")
        if errors:
            return None
        operation = data.get("bulkOperation")
        return operation.get("id") if operation else None

    async def _wait_for_bulk_operation(self, shop: str, token: str) -> Optional[Dict[str, Any]]:
        query = """
        query CurrentBulkOp {
          currentBulkOperation {
            id
            status
            url
            errorCode
            objectCount
            completedAt
            createdAt
          }
        }
        """
        for _ in range(60):
            response = await self._graphql_request(shop, token, {"query": query})
            operation = response.get("data", {}).get("currentBulkOperation")
            if not operation:
                await asyncio.sleep(2)
                continue

            status = operation.get("status")
            if status in ("COMPLETED", "FAILED", "CANCELED"):
                return operation
            await asyncio.sleep(2)
        return None

    async def _stream_bulk_file(self, url: str):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw = response.content
            try:
                decoded = gzip.decompress(raw)
            except OSError:
                decoded = raw

            for line in decoded.splitlines():
                if not line.strip():
                    continue
                yield json.loads(line.decode("utf-8"))

    async def _graphql_request(
        self, shop: str, token: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        endpoint = f"https://{shop}/admin/api/{self.api_version}/graphql.json"
        headers = self._graphql_headers(token)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize_from_graphql(
        self, record: Dict[str, Any], store: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        product_gid = self._gid_to_id(record.get("id"))
        images = self._extract_images(record.get("images"))
        variants = self._extract_variants_from_graph(record.get("variants"))
        return self._build_normalized_product(
            title=record.get("title"),
            description="",
            product_type=record.get("productType", ""),
            tags=record.get("tags") or [],
            status=record.get("status", "").lower(),
            vendor=record.get("vendor"),
            images=images,
            variants=variants,
            shopify_product_id=product_gid,
            store=store,
        )

    def _normalize_from_rest(
        self, product_data: Dict[str, Any], store: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        images = [img.get("src") for img in product_data.get("images", []) if img.get("src")]
        variants = self._extract_variants_from_rest(product_data.get("variants", []))
        return self._build_normalized_product(
            title=product_data.get("title"),
            description=product_data.get("body_html", "") or "",
            product_type=product_data.get("product_type", ""),
            tags=product_data.get("tags", "").split(",") if isinstance(product_data.get("tags"), str) else product_data.get("tags", []),
            status=product_data.get("status", "").lower(),
            vendor=product_data.get("vendor"),
            images=images,
            variants=variants,
            shopify_product_id=str(product_data.get("id")) if product_data.get("id") else None,
            store=store,
        )

    def _build_normalized_product(
        self,
        title: Optional[str],
        description: str,
        product_type: str,
        tags: List[str],
        status: str,
        vendor: Optional[str],
        images: List[str],
        variants: List[Dict[str, Any]],
        shopify_product_id: Optional[str],
        store: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not title or not variants:
            return None

        segment = self._classify_segment(title, tags, product_type)
        if not self._is_segment_allowed(segment, store):
            return None

        game = self._detect_game(title, product_type, tags)

        return {
            "title": title,
            "description": description,
            "segment": segment,
            "game": game,
            "status": status or "active",
            "vendor": vendor,
            "images": images,
            "variants": variants,
            "shopify_product_id": shopify_product_id,
        }

    def _extract_images(self, images_field: Optional[Dict[str, Any]]) -> List[str]:
        images: List[str] = []
        if not images_field:
            return images
        for edge in images_field.get("edges", []):
            node = edge.get("node") or {}
            url = node.get("url") or node.get("transformedSrc") or node.get("originalSrc")
            if url:
                images.append(url)
        return images

    def _extract_variants_from_graph(
        self, variants_field: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        variants: List[Dict[str, Any]] = []
        if not variants_field:
            return variants
        for edge in variants_field.get("edges", []):
            node = edge.get("node") or {}
            variant_id = self._gid_to_id(node.get("id"))
            if not variant_id:
                continue
            inventory_item = (node.get("inventoryItem") or {}).get("id")
            variants.append(
                {
                    "id": variant_id,
                    "sku": node.get("sku"),
                    "title": node.get("title"),
                    "price": float(node.get("price") or 0),
                    "inventory_quantity": int(node.get("inventoryQuantity") or 0),
                    "available": bool(node.get("availableForSale", True)),
                    "inventory_item_id": self._gid_to_id(inventory_item),
                }
            )
        return variants

    def _extract_variants_from_rest(
        self, variants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for variant in variants:
            variant_id = variant.get("id")
            if not variant_id:
                continue
            normalized.append(
                {
                    "id": str(variant_id),
                    "sku": variant.get("sku"),
                    "title": variant.get("title"),
                    "price": float(variant.get("price") or 0),
                    "inventory_quantity": int(variant.get("inventory_quantity") or 0),
                    "available": variant.get("inventory_quantity", 0) > 0,
                    "inventory_item_id": str(variant.get("inventory_item_id"))
                    if variant.get("inventory_item_id")
                    else None,
                }
            )
        return normalized

    # ------------------------------------------------------------------
    # Firestore persistence
    # ------------------------------------------------------------------
    def _upsert_product(self, normalized: Dict[str, Any]) -> str:
        products_query = (
            db.collection(PRODUCTS)
            .where("name", "==", normalized["title"])
            .limit(1)
            .stream()
        )
        product_id = None
        for doc in products_query:
            product_id = doc.id
            doc.reference.update(
                {
                    "updated_at": datetime.utcnow(),
                    "image_url": normalized["images"][0] if normalized["images"] else None,
                    "category": normalized["game"],
                    "segment": normalized["segment"],
                }
            )
            break

        if not product_id:
            ref = db.collection(PRODUCTS).document()
            ref.set(
                {
                    "name": normalized["title"],
                    "description": normalized.get("description"),
                    "category": normalized["game"],
                    "segment": normalized["segment"],
                    "image_url": normalized["images"][0] if normalized["images"] else None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "total_sales": 0,
                }
            )
            product_id = ref.id

        return product_id

    def _upsert_listings(
        self,
        store_data: Dict[str, Any],
        shop: str,
        product_id: str,
        normalized: Dict[str, Any],
    ):
        for variant in normalized["variants"]:
            listing_id = f"{shop}_{variant['id']}"
            listing_data = {
                "product_id": product_id,
                "product_name": normalized["title"],
                "product_segment": normalized["segment"],
                "product_game": normalized["game"],
                "store_id": shop,
                "store_name": store_data.get("store_name") or store_data.get("shop_domain"),
                "shopify_product_id": normalized.get("shopify_product_id"),
                "shopify_variant_id": variant["id"],
                "price": variant["price"],
                "quantity": max(0, variant["inventory_quantity"]),
                "inventory_item_id": variant.get("inventory_item_id"),
                "is_preorder": not variant["available"],
                "status": "active" if normalized["status"] == "active" else "inactive",
                "images": normalized["images"],
                "updated_at": datetime.utcnow(),
            }
            listing_ref = db.collection(SHOPIFY_LISTINGS).document(listing_id)
            existing = listing_ref.get()
            if existing.exists:
                listing_data["created_at"] = existing.to_dict().get("created_at")
            else:
                listing_data["created_at"] = datetime.utcnow()
            listing_ref.set(listing_data)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _classify_segment(self, title: str, tags: List[str], product_type: str) -> str:
        haystack = " ".join([title, product_type, *(tags or [])]).lower()
        if any(keyword in haystack for keyword in ["graded", "bgs", "psa", "cgc"]):
            return "graded"
        if any(keyword in haystack for keyword in ["single", "singles", "foil", "card -", "promo"]):
            return "singles"
        if any(keyword in haystack for keyword in ["sleeve", "deck box", "binder", "playmat", "accessor"]):
            return "accessories"
        return "sealed"

    def _detect_game(self, title: str, product_type: str, tags: List[str]) -> str:
        text = " ".join([title or "", product_type or "", *(tags or [])]).lower()
        mapping = {
            "pokémon": "Pokemon",
            "pokemon": "Pokemon",
            "yugioh": "Yu-Gi-Oh",
            "yu-gi-oh": "Yu-Gi-Oh",
            "mtg": "Magic: The Gathering",
            "magic": "Magic: The Gathering",
            "one piece": "One Piece",
            "lorcana": "Lorcana",
            "flesh and blood": "Flesh and Blood",
        }
        for needle, value in mapping.items():
            if needle in text:
                return value
        return "Other"

    def _is_segment_allowed(self, segment: str, store: Dict[str, Any]) -> bool:
        if segment == "sealed" or segment == "accessories":
            return True
        subscription_active = store.get("subscription_status") == "active"
        return subscription_active

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _rest_headers(self, token: str) -> Dict[str, str]:
        return {
            "X-Shopify-Access-Token": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _graphql_headers(self, token: str) -> Dict[str, str]:
        return {
            "X-Shopify-Access-Token": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _gid_to_id(self, gid: Optional[str]) -> Optional[str]:
        if not gid:
            return None
        if gid.isdigit():
            return gid
        return gid.split("/")[-1]
