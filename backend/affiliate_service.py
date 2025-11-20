"""
Amazon.ca affiliate ingestion and pricing utilities.
"""
import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from database import AFFILIATE_PRODUCTS, PRODUCTS, db

logger = logging.getLogger(__name__)

AMAZON_TCG_QUERIES: List[Tuple[str, str]] = [
    ("Pokemon TCG sealed booster box", "Pokemon"),
    ("Magic The Gathering sealed booster", "Magic: The Gathering"),
    ("Yu-Gi-Oh sealed booster box", "Yu-Gi-Oh"),
    ("One Piece TCG sealed", "One Piece"),
    ("Disney Lorcana sealed booster", "Lorcana"),
    ("Flesh and Blood sealed booster", "Flesh and Blood"),
    ("Digimon sealed booster box", "Digimon"),
    ("Weiss Schwarz sealed booster", "Weiss Schwarz"),
    ("Dragon Ball Super sealed booster", "Dragon Ball Super"),
    ("Final Fantasy TCG sealed booster", "Final Fantasy TCG"),
    ("Cardfight Vanguard sealed booster", "Cardfight Vanguard"),
]


class AffiliateService:
    """Handle Amazon.ca affiliate integrations."""

    def __init__(self):
        self.amazon_tag = (
            os.getenv("AMAZON_CA_AFFILIATE_TAG")
            or os.getenv("AMAZON_ASSOCIATE_TAG")
            or "geocheapest-20"
        )
        self.amazon_api_key = os.getenv("RAPIDAPI_KEY")
        self.amazon_host = os.getenv(
            "RAPIDAPI_AMAZON_HOST", "amazon-data-scraper126.p.rapidapi.com"
        )
        self.amazon_min_rating = float(os.getenv("AMAZON_MIN_RATING", "4.0"))
        self.amazon_min_reviews = int(os.getenv("AMAZON_MIN_REVIEWS", "50"))
        self.amazon_sync_interval = int(os.getenv("AMAZON_SYNC_INTERVAL_SECONDS", "86400"))
        self.amazon_sync_interval = int(os.getenv("AMAZON_SYNC_INTERVAL_SECONDS", "86400"))
        self.amazon_sync_enabled = bool(self.amazon_api_key)
        
        # Vetted Vendors (Can be moved to DB/Env later)
        self.vetted_vendors = [
            "Amazon.ca",
            "Amazon",
            "401 Games",
            "Hobbiesville",
            "Dolly's Toys and Games",
            "Game Shack",
            "Red Nails II"
        ]

    # ------------------------------------------------------------------
    # Amazon helpers
    # ------------------------------------------------------------------
    async def search_amazon_product(self, search_query: str) -> List[Dict[str, Any]]:
        """Search Amazon.ca for products using RapidAPI."""
        if not self.amazon_api_key:
            return []

        url = f"https://{self.amazon_host}/search"
        headers = {
            "X-RapidAPI-Key": self.amazon_api_key,
            "X-RapidAPI-Host": self.amazon_host,
        }
        params = {"query": search_query, "country": "CA"}

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("results", data.get("items", []))
                logger.warning("Amazon search failed (%s): %s", response.status_code, response.text[:200])
        except Exception as exc:
            logger.error("Amazon search error: %s", exc)
        return []

    async def get_amazon_product_details(self, asin: str) -> Optional[Dict[str, Any]]:
        if not self.amazon_api_key:
            return None
        url = f"https://{self.amazon_host}/product/{asin}"
        headers = {
            "X-RapidAPI-Key": self.amazon_api_key,
            "X-RapidAPI-Host": self.amazon_host,
        }
        params = {"country": "CA"}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    return response.json()
        except Exception as exc:
            logger.error("Amazon details error: %s", exc)
        return None

    def build_amazon_affiliate_url(self, asin: str) -> str:
        return f"https://www.amazon.ca/dp/{asin}?tag={self.amazon_tag}"

    async def sync_amazon_tcg_products(self) -> int:
        """Pull sealed TCG inventory for every supported game."""
        if not self.amazon_sync_enabled:
            return 0

        total = 0
        for query, game in AMAZON_TCG_QUERIES:
            results = await self.search_amazon_product(query)
            for result in results:
                normalized = self._normalize_amazon_result(result, game)
                if not normalized:
                    continue
                product_id = self._upsert_product(normalized)
                self._upsert_amazon_listing(product_id, normalized)
                total += 1
        logger.info("Amazon sync completed with %s listings", total)
        return total

    def _normalize_amazon_result(
        self, result: Dict[str, Any], game: str
    ) -> Optional[Dict[str, Any]]:
        asin = result.get("asin") or result.get("asin_id") or result.get("product_id")
        if not asin:
            return None

        rating = float(result.get("rating") or result.get("stars") or 0)
        review_count = int(result.get("reviews_count") or result.get("reviews") or 0)
        if rating < self.amazon_min_rating or review_count < self.amazon_min_reviews:
            return None

        price_raw = (
            result.get("price")
            or result.get("price_current")
            or result.get("price_string")
            or ""
        )
        price = self._parse_price(price_raw)
        if price <= 0:
            return None

        availability = result.get("availability", True)
        in_stock = (
            result.get("in_stock", True)
            if isinstance(result.get("in_stock"), bool)
            else "out of stock" not in str(availability).lower()
        )
        if not in_stock:
            return None

        title = result.get("title") or result.get("name")
        if not title:
            return None

        image = (
            result.get("image")
            or result.get("thumbnail")
            or (result.get("images") or [None])[0]
        )
        description = result.get("description") or result.get("snippet") or ""
        product_type = self._classify_product_type(title)
        release_status = self._classify_release_status(title, availability)
        seller = result.get("merchant") or result.get("seller_name") or "Amazon Marketplace"
        
        # Vetted Vendor Check
        if not any(v.lower() in seller.lower() for v in self.vetted_vendors):
            # Optional: Log skipped vendor for review
            # logger.debug(f"Skipping non-vetted vendor: {seller}")
            return None

        return {
            "asin": asin,
            "title": title,
            "description": description,
            "price": price,
            "image": image,
            "url": result.get("url") or f"https://www.amazon.ca/dp/{asin}",
            "rating": rating,
            "review_count": review_count,
            "prime": bool(result.get("is_prime") or result.get("amazonPrime")),
            "game": game,
            "product_type": product_type,
            "release_status": release_status,
            "seller": seller,
            "quantity": int(result.get("stock_count") or 50),
            "in_stock": in_stock,
        }

    def _parse_price(self, price_str: str) -> float:
        if not price_str:
            return 0.0
        cleaned = re.sub(r"[^0-9\.]", "", str(price_str))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _classify_product_type(self, title: str) -> str:
        lowered = title.lower()
        if "booster" in lowered:
            return "booster_box"
        if "elite trainer" in lowered or "etb" in lowered:
            return "elite_trainer_box"
        if "starter" in lowered or "structure deck" in lowered:
            return "starter_deck"
        if "bundle" in lowered or "gift" in lowered:
            return "bundle"
        if "collector" in lowered:
            return "collector_box"
        if "case" in lowered:
            return "case"
        return "sealed_product"

    def _classify_release_status(self, title: str, availability: Any) -> str:
        title_lower = title.lower()
        if "pre-order" in title_lower or "preorder" in title_lower:
            return "preorder"
        if "new release" in title_lower:
            return "new_release"
        if isinstance(availability, str) and "best seller" in availability.lower():
            return "best_seller"
        return "available"

    def _upsert_product(self, normalized: Dict[str, Any]) -> str:
        query = (
            db.collection(PRODUCTS)
            .where("name", "==", normalized["title"])
            .limit(1)
            .stream()
        )
        product_id = None
        for doc in query:
            product_id = doc.id
            doc.reference.update(
                {
                    "updated_at": datetime.utcnow(),
                    "image_url": normalized["image"],
                    "category": normalized["game"],
                    "segment": "sealed",
                }
            )
            break

        if not product_id:
            ref = db.collection(PRODUCTS).document()
            ref.set(
                {
                    "name": normalized["title"],
                    "description": normalized["description"],
                    "category": normalized["game"],
                    "segment": "sealed",
                    "image_url": normalized["image"],
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "total_sales": 0,
                }
            )
            product_id = ref.id

        return product_id

    def _upsert_amazon_listing(self, product_id: str, normalized: Dict[str, Any]):
        doc_id = f"amazon_{normalized['asin']}"
        affiliate_url = self.build_amazon_affiliate_url(normalized["asin"])
        data = {
            "product_id": product_id,
            "affiliate_name": "Amazon.ca",
            "affiliate_url": affiliate_url,
            "asin": normalized["asin"],
            "price": normalized["price"],
            "in_stock": normalized["in_stock"],
            "quantity": normalized["quantity"],
            "estimated_shipping": 0.0,
            "commission_rate": 0.05,
            "source": "amazon",
            "game": normalized["game"],
            "product_type": normalized["product_type"],
            "release_status": normalized["release_status"],
            "vendor_rating": normalized["rating"],
            "review_count": normalized["review_count"],
            "is_prime": normalized["prime"],
            "images": [normalized["image"]] if normalized["image"] else [],
            "description": normalized["description"],
            "status": "active",
            "updated_at": datetime.utcnow(),
        }
        listing_ref = db.collection(AFFILIATE_PRODUCTS).document(doc_id)
        existing = listing_ref.get()
        if existing.exists:
            data["created_at"] = existing.to_dict().get("created_at")
        else:
            data["created_at"] = datetime.utcnow()
        listing_ref.set(data)

    # ------------------------------------------------------------------
    # Price refresh
    # ------------------------------------------------------------------
    async def update_affiliate_prices(self):
        """Lightweight price refresh for Amazon listings."""
        amazon_docs = db.collection(AFFILIATE_PRODUCTS).where(
            "affiliate_name", "==", "Amazon.ca"
        ).stream()
        for doc in amazon_docs:
            listing = doc.to_dict()
            asin = listing.get("asin")
            if not asin:
                continue
            details = await self.get_amazon_product_details(asin)
            if details:
                price = self._parse_price(details.get("price") or "")
                stock = details.get("in_stock", True)
                doc.reference.update(
                    {
                        "price": price or listing.get("price", 0),
                        "in_stock": bool(stock),
                        "updated_at": datetime.utcnow(),
                    }
                )


async def amazon_sync_loop(service: AffiliateService):
    """Background cron loop to keep Amazon listings fresh."""
    if not service.amazon_sync_enabled:
        return
    delay = max(3600, service.amazon_sync_interval)
    while True:
        try:
            await service.sync_amazon_tcg_products()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Amazon sync loop error: %s", exc)
        await asyncio.sleep(delay)
