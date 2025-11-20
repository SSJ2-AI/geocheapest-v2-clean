"""
Shippo integration for rates, labels, and tracking.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class ShippoService:
    """Handle Shippo API for real-time rates and label generation."""

    def __init__(self):
        self.api_token = os.getenv("SHIPPO_API_KEY")
        self.base_url = "https://api.goshippo.com"
        self.default_from = {
            "name": os.getenv("SHIPPO_FROM_NAME", "GeoCheapest Vendor"),
            "street1": os.getenv("SHIPPO_FROM_STREET", "123 Front St W"),
            "city": os.getenv("SHIPPO_FROM_CITY", "Toronto"),
            "state": os.getenv("SHIPPO_FROM_PROVINCE", "ON"),
            "zip": os.getenv("SHIPPO_FROM_POSTAL", "M5H 2N2"),
            "country": os.getenv("SHIPPO_FROM_COUNTRY", "CA"),
            "phone": os.getenv("SHIPPO_FROM_PHONE", "5555555555"),
        }
        supported = os.getenv(
            "SHIPPO_SUPPORTED_PROVIDERS", "canada_post,ups,fedex,purolator"
        )
        self.supported_providers = [
            provider.strip().lower() for provider in supported.split(",") if provider.strip()
        ]
        self.carrier_accounts = [
            acct.strip()
            for acct in os.getenv("SHIPPO_CARRIER_ACCOUNTS", "").split(",")
            if acct.strip()
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def calculate_shipping(
        self,
        store_id: str,
        product_id: str,
        quantity: int,
        shipping_address: Dict[str, str],
    ) -> float:
        """Return the cheapest Shippo rate or fallback estimate."""
        if not self.api_token:
            return self.estimate_shipping(shipping_address, quantity)

        shipment = await self._create_shipment(
            shipping_address, quantity, reference=f"{store_id}:{product_id}"
        )
        rates = shipment.get("rates", [])
        if not rates:
            return self.estimate_shipping(shipping_address, quantity)

        filtered = [
            rate
            for rate in rates
            if rate.get("provider", "").lower() in self.supported_providers
            and rate.get("currency") == "CAD"
        ]
        candidates = filtered or rates
        best = min(candidates, key=lambda r: float(r.get("amount_local", r.get("amount", 0))))
        try:
            return float(best.get("amount_local", best.get("amount", 0)))
        except (TypeError, ValueError):
            return self.estimate_shipping(shipping_address, quantity)

    def estimate_shipping(self, shipping_address: Dict[str, str], quantity: int) -> float:
        """Graceful fallback if Shippo is unavailable."""
        base_rate = 11.0
        per_item = 3.5
        province = shipping_address.get("province", "ON")
        multipliers = {
            "ON": 1.0,
            "QC": 1.1,
            "BC": 1.3,
            "AB": 1.2,
            "MB": 1.2,
            "SK": 1.2,
            "NS": 1.3,
            "NB": 1.3,
            "NL": 1.5,
            "PE": 1.4,
            "NT": 2.0,
            "YT": 2.0,
            "NU": 2.5,
        }
        multiplier = multipliers.get(province, 1.2)
        return (base_rate + per_item * max(quantity - 1, 0)) * multiplier

    async def create_shipping_label(
        self,
        order_id: str,
        shipping_address: Dict[str, Any],
        items: List[Dict[str, Any]],
        is_return: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Generate a Shippo label and return metadata."""
        if not self.api_token:
            return None

        shipment = await self._create_shipment(
            shipping_address,
            max(len(items), 1),
            reference=order_id,
            is_return=is_return,
            items=items,
        )
        rates = shipment.get("rates", [])
        if not rates:
            return None
        rate = self._select_rate(rates)
        if not rate:
            return None

        transaction = await self._request(
            "POST",
            "/transactions/",
            json={
                "rate": rate["object_id"],
                "label_file_type": "PDF",
                "async": False,
            },
        )
        if transaction.get("status") != "SUCCESS":
            logger.warning("Shippo label failed: %s", transaction)
            return None

        return {
            "transaction_id": transaction.get("object_id"),
            "tracking_number": transaction.get("tracking_number"),
            "label_url": transaction.get("label_url"),
            "carrier": rate.get("provider"),
            "service_level": rate.get("servicelevel", {}).get("name"),
            "cost": float(rate.get("amount_local", rate.get("amount", 0))),
            "currency": rate.get("currency", "CAD"),
        }

    async def create_return_label(
        self,
        order_id: str,
        original_address: Dict[str, Any],
        items: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Reverse the shipment direction for returns."""
        customer_service = {
            "name": self.default_from["name"],
            "street": self.default_from["street1"],
            "city": self.default_from["city"],
            "province": self.default_from["state"],
            "postal_code": self.default_from["zip"],
            "country": self.default_from["country"],
            "phone": self.default_from["phone"],
        }
        return await self.create_shipping_label(
            order_id=f"{order_id}-return",
            shipping_address=customer_service,
            items=items,
            is_return=True,
        )

    async def track_shipment(self, carrier: str, tracking_number: str) -> Optional[Dict[str, Any]]:
        """Get tracking information via Shippo."""
        if not self.api_token:
            return None
        path = f"/tracks/{carrier}/{tracking_number}/"
        return await self._request("GET", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _create_shipment(
        self,
        destination: Dict[str, Any],
        quantity: int,
        reference: Optional[str] = None,
        is_return: bool = False,
        items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "address_to": self._build_address(destination),
            "address_from": self._build_address(self.default_from),
            "parcels": [self._build_parcel(quantity, items)],
            "async": False,
            "reference": reference,
        }
        if self.carrier_accounts:
            payload["carrier_accounts"] = self.carrier_accounts
        if is_return:
            payload["metadata"] = "return_label"
        return await self._request("POST", "/shipments/", json=payload)

    def _build_address(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": data.get("name"),
            "street1": data.get("street1") or data.get("street"),
            "city": data.get("city"),
            "state": data.get("state") or data.get("province"),
            "zip": data.get("zip") or data.get("postal_code"),
            "country": data.get("country", "CA"),
            "phone": data.get("phone"),
        }

    def _build_parcel(self, quantity: int, items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        weight_per_item = 0.9  # kg per sealed product approx 2 lbs
        height_per_item = 4
        return {
            "length": 8,
            "width": 6,
            "height": max(4, height_per_item * quantity),
            "distance_unit": "in",
            "weight": max(0.5, weight_per_item * quantity),
            "mass_unit": "lb",
            "metadata": f"{len(items or [])} items" if items else None,
        }

    def _select_rate(self, rates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        preferred = [
            rate
            for rate in rates
            if rate.get("provider", "").lower() in self.supported_providers
            and rate.get("currency") == "CAD"
        ]
        candidates = preferred or rates
        if not candidates:
            return None
        return min(candidates, key=lambda r: float(r.get("amount_local", r.get("amount", 0))))

    async def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_token:
            return {}
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"ShippoToken {self.api_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, headers=headers, json=json)
            if response.status_code >= 400:
                logger.warning("Shippo request %s failed: %s %s", path, response.status_code, response.text[:200])
                return {}
            try:
                return response.json()
            except ValueError:
                logger.error("Shippo returned invalid JSON for %s", path)
                return {}
