"""
Stripe integration service for payments and payouts
"""
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import stripe

from database import (
    db,
    SHOPIFY_LISTINGS,
    STORES,
    PRODUCTS,
    SELLER_PAYOUTS,
    USERS,
)

logger = logging.getLogger(__name__)


class StripeService:
    """Handle Stripe payments, Connect payouts, and vendor billing"""

    def __init__(self):
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        self.sealed_categories = {
            "pokemon",
            "magic",
            "yu-gi-oh",
            "yu gi oh",
            "one piece",
            "lorcana",
            "flesh and blood",
        }
        self.platform_fee_sealed = Decimal(
            os.getenv("STRIPE_PLATFORM_COMMISSION_SEALED", "0.045")
        )
        self.platform_fee_singles = Decimal(
            os.getenv("STRIPE_PLATFORM_COMMISSION_SINGLES", "0.02")
        )
        self.percent_processing_fee = Decimal(
            os.getenv("STRIPE_CARD_PERCENT_FEE", "0.029")
        )
        self.fixed_processing_fee = Decimal(
            os.getenv("STRIPE_CARD_FIXED_FEE", "0.30")
        )
        self.subscription_price_ids = {
            "basic": os.getenv("STRIPE_VENDOR_SUB_BASIC_PRICE_ID"),
            "growth": os.getenv("STRIPE_VENDOR_SUB_GROWTH_PRICE_ID"),
            "pro": os.getenv("STRIPE_VENDOR_SUB_PRO_PRICE_ID"),
        }

    async def create_connect_account(
        self,
        shop: str,
        vendor_email: Optional[str] = None,
        business_name: Optional[str] = None,
        business_url: Optional[str] = None,
    ) -> Optional[str]:
        """Create Stripe Connect account for vendor"""
        email = vendor_email or f"owner@{shop}"
        profile_name = business_name or shop
        profile_url = business_url or f"https://{shop}"

        try:
            account = stripe.Account.create(
                type="express",
                country="CA",
                email=email,
                capabilities={
                    "card_payments": {"requested": True},
                    "transfers": {"requested": True},
                },
                business_type="company",
                metadata={"shop_domain": shop},
                business_profile={
                    "name": profile_name,
                    "url": profile_url,
                    "mcc": "5945",  # Hobby, toy, and game shops
                    "support_email": email,
                },
            )
            return account.id
        except stripe.error.StripeError as exc:
            logger.error("Stripe Connect error for %s: %s", shop, exc)
            return None

    def get_commission_rate_for_tier(self, tier: str, product_type: str = "sealed") -> float:
        """
        Dynamic Pricing Model:
        - Sealed Products: Free listing (Standard Commission)
        - Singles/Graded: Premium listing (Higher Commission or Monthly Fee required)
        
        Rates (Commission %):
        """
        # Base rates
        rates = {
            "basic": 0.10,
            "growth": 0.08,
            "pro": 0.06
        }
        base_rate = rates.get(tier, 0.10)
        
        # Category Surcharge
        # Singles/Graded require more verification/handling, so we might charge more
        # OR: If the user wants to charge for uploading them, that's a subscription gate check
        if product_type in ["single", "graded"]:
            # Example: 2% surcharge for singles if on Basic tier
            if tier == "basic":
                return base_rate + 0.02
                
        return base_rate

    async def create_account_link(
        self,
        account_id: str,
        refresh_url: Optional[str] = None,
        return_url: Optional[str] = None,
    ) -> Optional[str]:
        """Generate onboarding link for vendor Connect account"""
        try:
            link = stripe.AccountLink.create(
                account=account_id,
                refresh_url=refresh_url or f"{self.frontend_url}/vendor/connect/retry",
                return_url=return_url or f"{self.frontend_url}/vendor/dashboard",
                type="account_onboarding",
            )
            return link.url
        except stripe.error.StripeError as exc:
            logger.error("Stripe AccountLink error: %s", exc)
            return None

    async def ensure_platform_customer(self, user_id: str, email: str) -> str:
        """Create or re-use a platform Stripe Customer for registered users"""
        user_ref = db.collection(USERS).document(user_id)
        doc = user_ref.get()
        user_data = doc.to_dict() if doc.exists else {}
        customer_id = user_data.get("stripe_customer_id")

        if not customer_id:
            customer = stripe.Customer.create(
                email=email,
                metadata={"user_id": user_id, "type": "buyer"},
            )
            customer_id = customer.id
            user_ref.set(
                {
                    "email": email,
                    "stripe_customer_id": customer_id,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                },
                merge=True,
            )
        else:
            if email and user_data.get("email") != email:
                stripe.Customer.modify(customer_id, email=email)
                user_ref.update({"email": email, "updated_at": datetime.utcnow()})

        return customer_id

    async def create_setup_intent_for_user(
        self,
        user_id: str,
        email: str,
    ) -> Dict[str, str]:
        """Create SetupIntent so registered users can save payment methods"""
        customer_id = await self.ensure_platform_customer(user_id, email)
        setup_intent = stripe.SetupIntent.create(
            customer=customer_id,
            payment_method_types=["card"],
            usage="off_session",
        )
        return {
            "customer_id": customer_id,
            "client_secret": setup_intent.client_secret,
        }

    async def create_checkout_session(
        self,
        items: List[Dict[str, Any]],
        customer_email: str,
        shipping_address: Dict[str, Any],
        user_id: Optional[str] = None,
        save_payment_method: bool = False,
    ):
        """Create Stripe Checkout session for multi-vendor carts"""
        if not items:
            raise ValueError("At least one line item is required")
        shipping_address = shipping_address or {}

        transfer_group = f"order-{uuid4()}"
        line_items: List[Dict[str, Any]] = []
        metadata_items: List[Dict[str, Any]] = []
        shipping_totals = defaultdict(lambda: Decimal("0"))
        store_names: Dict[str, str] = {}
        platform_commission_total = Decimal("0")
        order_gross_total = Decimal("0")

        for item in items:
            listing_id = item.get("listing_id")
            product_id = item.get("product_id")
            quantity = int(item.get("quantity", 1))
            if not listing_id or not product_id:
                continue

            listing_doc = (
                db.collection(SHOPIFY_LISTINGS).document(listing_id).get()
            )
            if not listing_doc.exists:
                continue
            listing = listing_doc.to_dict()

            store_id = item.get("store_id") or listing.get("store_id")
            if not store_id:
                continue

            store_doc = db.collection(STORES).document(store_id).get()
            if not store_doc.exists:
                continue
            store = store_doc.to_dict()
            stripe_account_id = store.get("stripe_account_id")
            if not stripe_account_id:
                raise RuntimeError(f"Store {store_id} is missing Stripe Connect details")

            product_doc = db.collection(PRODUCTS).document(product_id).get()
            if not product_doc.exists:
                continue
            product = product_doc.to_dict()

            unit_price = self._to_decimal(listing.get("price"))
            product_total = unit_price * Decimal(quantity)
            shipping_cost = self._to_decimal(item.get("shipping_cost"))
            line_total = product_total + shipping_cost

            commission_rate = self._determine_commission_rate(product)
            platform_commission = (product_total * commission_rate).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
            platform_commission_total += platform_commission
            order_gross_total += line_total
            shipping_totals[store_id] += shipping_cost
            store_names[store_id] = store.get("store_name") or store.get(
                "shop_domain"
            ) or store_id

            images = [product.get("image_url")] if product.get("image_url") else []
            line_items.append(
                {
                    "price_data": {
                        "currency": "cad",
                        "unit_amount": self._decimal_to_cents(unit_price),
                        "product_data": {
                            "name": product.get("name", "TCG Product"),
                            "description": f"Sold by {store_names[store_id]}",
                            "images": images,
                        },
                    },
                    "quantity": quantity,
                }
            )

            metadata_items.append(
                {
                    "store_id": store_id,
                    "store_name": store_names[store_id],
                    "stripe_account_id": stripe_account_id,
                    "listing_id": listing_id,
                    "product_id": product_id,
                    "quantity": quantity,
                    "product_total": str(product_total),
                    "shipping_total": str(shipping_cost),
                    "gross_total": str(line_total),
                    "commission_rate": str(commission_rate),
                    "platform_commission": str(platform_commission),
                    "category": product.get("category"),
                }
            )

        for store_id, shipping_amount in shipping_totals.items():
            if shipping_amount <= 0:
                continue
            line_items.append(
                {
                    "price_data": {
                        "currency": "cad",
                        "unit_amount": self._decimal_to_cents(shipping_amount),
                        "product_data": {
                            "name": f"Shipping - {store_names.get(store_id, 'Vendor')}",
                            "description": "Vendor-fulfilled shipping",
                        },
                    },
                    "quantity": 1,
                }
            )

        metadata_payload = {
            "items": json.dumps(metadata_items, separators=(",", ":")),
            "shipping_address": json.dumps(shipping_address, separators=(",", ":")),
            "transfer_group": transfer_group,
            "checkout_type": "shopify_order",
            "platform_commission_total": str(platform_commission_total),
            "order_gross_total": str(order_gross_total),
            "user_id": user_id or "",
        }

        payment_intent_data: Dict[str, Any] = {
            "transfer_group": transfer_group,
            "metadata": metadata_payload,
        }

        if save_payment_method and user_id:
            payment_intent_data["setup_future_usage"] = "off_session"

        checkout_payload: Dict[str, Any] = {
            "payment_method_types": ["card"],
            "line_items": line_items,
            "mode": "payment",
            "success_url": f"{self.frontend_url}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{self.frontend_url}/checkout/cancel",
            "metadata": metadata_payload,
            "payment_intent_data": payment_intent_data,
            "automatic_tax": {"enabled": True},
        }

        if user_id:
            customer_id = await self.ensure_platform_customer(user_id, customer_email)
            checkout_payload["customer"] = customer_id
        else:
            checkout_payload["customer_email"] = customer_email

        session = stripe.checkout.Session.create(**checkout_payload)
        return session

    async def process_commission(
        self,
        order_id: str,
        metadata: Dict[str, Any],
        payment_intent: Optional[stripe.PaymentIntent]
    ):
        """
        Calculate and record commissions based on vendor tiers
        """
        # Logic to fetch vendor tier and apply dynamic rate
        # This is a simplified placeholder for the complex logic
        pass
        items_raw = metadata.get("items")
        if not items_raw:
            logger.warning("Stripe metadata missing items for order %s", order_id)
            return

        try:
            items = json.loads(items_raw)
        except json.JSONDecodeError:
            logger.error("Invalid Stripe metadata JSON for order %s", order_id)
            return

        vendor_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "product_total": Decimal("0"),
                "shipping_total": Decimal("0"),
                "platform_commission": Decimal("0"),
                "stripe_account_id": None,
                "store_name": "",
                "items": [],
            }
        )

        for item in items:
            store_id = item.get("store_id")
            if not store_id:
                continue
            vendor = vendor_totals[store_id]
            vendor["items"].append(item)
            vendor["product_total"] += self._to_decimal(item.get("product_total"))
            vendor["shipping_total"] += self._to_decimal(item.get("shipping_total"))
            vendor["platform_commission"] += self._to_decimal(
                item.get("platform_commission")
            )
            vendor["stripe_account_id"] = item.get("stripe_account_id")
            vendor["store_name"] = item.get("store_name", store_id)

        order_gross_total = self._to_decimal(
            metadata.get("order_gross_total")
        ) or sum(
            vendor["product_total"] + vendor["shipping_total"]
            for vendor in vendor_totals.values()
        )
        total_processing_fee = self.calculate_total_stripe_fee(
            payment_intent, order_gross_total
        )

        for store_id, vendor in vendor_totals.items():
            gross = vendor["product_total"] + vendor["shipping_total"]
            if gross <= 0:
                continue

            fee_share = (
                (gross / order_gross_total) * total_processing_fee
                if order_gross_total > 0
                else Decimal("0")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            vendor_payout = gross - vendor["platform_commission"] - fee_share
            if vendor_payout <= 0:
                logger.warning(
                    "Net payout for store %s is non-positive (%.2f)", store_id, vendor_payout
                )
                continue

            transfer_id = await self.transfer_to_vendor(
                vendor.get("stripe_account_id"),
                vendor_payout,
                metadata.get("transfer_group") or order_id,
            )

            payout_data = {
                "store_id": store_id,
                "order_id": order_id,
                "amount": float(gross),
                "commission_amount": float(vendor["platform_commission"]),
                "stripe_fee": float(fee_share),
                "net_payout": float(vendor_payout),
                "status": "processing" if transfer_id else "pending",
                "stripe_transfer_id": transfer_id,
                "created_at": datetime.utcnow(),
            }
            db.collection(SELLER_PAYOUTS).document().set(payout_data)

    async def transfer_to_vendor(
        self,
        stripe_account_id: Optional[str],
        amount: Decimal,
        transfer_group: str,
    ) -> Optional[str]:
        """Transfer funds to vendor via Stripe Connect"""
        if not stripe_account_id or amount <= 0:
            return None

        try:
            transfer = stripe.Transfer.create(
                amount=self._decimal_to_cents(amount),
                currency="cad",
                destination=stripe_account_id,
                transfer_group=transfer_group,
            )
            return transfer.id
        except stripe.error.StripeError as exc:
            logger.error(
                "Stripe transfer error to %s: %s", stripe_account_id, exc
            )
            return None

    async def process_refund(
        self,
        payment_intent_id: str,
        items: List[Dict[str, Any]],
    ):
        """Process refund for returned items"""
        try:
            refund_amount = sum(item["total_price"] for item in items)
            refund = stripe.Refund.create(
                payment_intent=payment_intent_id,
                amount=self._decimal_to_cents(Decimal(refund_amount)),
            )

            order_ids = {item.get("order_id") for item in items if item.get("order_id")}
            for order_id in order_ids:
                payout_docs = (
                    db.collection(SELLER_PAYOUTS)
                    .where("order_id", "==", order_id)
                    .stream()
                )
                for doc in payout_docs:
                    doc.reference.update(
                        {"status": "reversed", "reversed_at": datetime.utcnow()}
                    )

            return refund.id
        except stripe.error.StripeError as exc:
            logger.error("Refund error: %s", exc)
            return None

    async def create_vendor_subscription_checkout(
        self,
        store_id: str,
        store: Dict[str, Any],
        tier: str,
        contact_email: str,
        success_url: Optional[str] = None,
        cancel_url: Optional[str] = None,
    ) -> Tuple[stripe.checkout.Session, str]:
        """Create a Checkout Session for vendor subscription billing"""
        price_id = self.subscription_price_ids.get(tier)
        if not price_id:
            raise ValueError(f"No Stripe Price configured for tier '{tier}'")

        billing_customer_id = store.get("stripe_billing_customer_id")
        if not billing_customer_id:
            customer = stripe.Customer.create(
                email=contact_email,
                metadata={"store_id": store_id, "type": "vendor_billing"},
            )
            billing_customer_id = customer.id

        session = stripe.checkout.Session.create(
            customer=billing_customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url
            or f"{self.frontend_url}/vendor/subscription/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=cancel_url
            or f"{self.frontend_url}/vendor/subscription/cancel",
            allow_promotion_codes=True,
            subscription_data={
                "metadata": {"store_id": store_id, "subscription_tier": tier},
            },
            metadata={
                "store_id": store_id,
                "subscription_tier": tier,
                "checkout_type": "vendor_subscription",
            },
        )

        return session, billing_customer_id

    async def create_vendor_billing_portal_session(
        self,
        store: Dict[str, Any],
        return_url: Optional[str] = None,
    ) -> stripe.billing_portal.Session:
        """Allow vendor to manage subscription via Stripe billing portal"""
        billing_customer_id = store.get("stripe_billing_customer_id")
        if not billing_customer_id:
            raise RuntimeError("Vendor is not subscribed yet")

        session = stripe.billing_portal.Session.create(
            customer=billing_customer_id,
            return_url=return_url or f"{self.frontend_url}/vendor/dashboard",
        )
        return session

    def calculate_total_stripe_fee(
        self,
        payment_intent: Optional[Dict[str, Any]],
        amount: Decimal,
    ) -> Decimal:
        """Determine Stripe processing fee from balance transaction or fallback calculation"""
        if payment_intent:
            latest_charge = payment_intent.get("latest_charge")
            if latest_charge and isinstance(latest_charge, dict):
                balance_txn = latest_charge.get("balance_transaction")
                if isinstance(balance_txn, dict):
                    fee = Decimal(balance_txn.get("fee", 0)) / Decimal("100")
                    return fee.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        fee = (amount * self.percent_processing_fee) + self.fixed_processing_fee
        return fee.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _determine_commission_rate(self, product: Dict[str, Any]) -> Decimal:
        category = (product.get("category") or "").lower()
        if category in self.sealed_categories:
            return self.platform_fee_sealed
        return self.platform_fee_singles

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        return Decimal(str(value))

    @staticmethod
    def _decimal_to_cents(value: Decimal) -> int:
        return int(
            (value * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
