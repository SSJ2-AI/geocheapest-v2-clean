"""
Pydantic models for GeoCheapest v2
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class StoreStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class ReturnStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class Condition(str, Enum):
    NM = "Near Mint"
    LP = "Lightly Played"
    MP = "Moderately Played"
    HP = "Heavily Played"
    DMG = "Damaged"
    SEALED = "Sealed"

class GradingInfo(BaseModel):
    grader: str  # PSA, BGS, CGC, etc.
    grade: float
    certification_number: str
    population: Optional[int] = None


class UserPreference(BaseModel):
    favorite_games: List[str] = [] # e.g. ["Pokemon", "Magic"]
    intent_tags: List[str] = [] # e.g. ["investor", "player", "collector"]
    budget_tier: str = "medium" # low, medium, high
    last_active: datetime = datetime.utcnow()


# ==================== AUTH MODELS ====================

class User(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    is_vendor: bool = False
    preferences: Optional[UserPreference] = None
    created_at: datetime = datetime.utcnow()

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None



# ==================== STORE MODELS ====================

class Store(BaseModel):
    shop_domain: str
    access_token: str
    stripe_account_id: str
    status: StoreStatus
    created_at: datetime
    last_sync_at: Optional[datetime] = None
    total_products: int = 0
    total_sales: float = 0.0
    commission_rate: float = 0.10


# ==================== PRODUCT MODELS ====================

class Product(BaseModel):
    """Canonical product in the system"""
    name: str
    description: Optional[str] = None
    category: str  # "Pokemon", "Yu-Gi-Oh", "Magic", "One Piece", etc.
    image_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    total_sales: int = 0


class ShopifyListing(BaseModel):
    """Product listing from Shopify store"""
    product_id: str  # References Product
    store_id: str  # Shop domain
    store_name: str
    shopify_product_id: str
    shopify_variant_id: str
    shopify_variant_id: str
    price: float
    quantity: int
    condition: Condition = Condition.SEALED
    grading: Optional[GradingInfo] = None
    is_first_edition: bool = False
    is_preorder: bool = False
    preorder_release_date: Optional[datetime] = None
    status: str = "active"
    created_at: datetime
    updated_at: datetime


class AffiliateProduct(BaseModel):
    """Product from affiliate (Amazon.ca / other integrations)"""
    product_id: str  # References Product
    affiliate_name: str  # "Amazon.ca" or other affiliate source
    affiliate_url: str  # URL with affiliate tag
    price: float
    in_stock: bool
    estimated_shipping: float = 10.0
    status: str = "active"
    created_at: datetime
    updated_at: datetime


# ==================== ORDER MODELS ====================

class Order(BaseModel):
    customer_email: EmailStr
    stripe_session_id: Optional[str] = None
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    total_amount: float
    total_shipping: float
    platform_commission: float
    stripe_fee: float
    currency: str = "CAD"
    shipping_address: Dict[str, str]


class OrderItem(BaseModel):
    order_id: str
    product_id: str
    listing_id: str
    source: str  # "shopify" or "affiliate"
    store_id: Optional[str] = None  # For Shopify orders
    quantity: int
    unit_price: float
    total_price: float
    commission_rate: float
    commission_amount: float
    vendor_payout: float
    status: str
    tracking_number: Optional[str] = None


# ==================== PAYOUT MODELS ====================

class SellerPayout(BaseModel):
    store_id: str
    order_id: str
    amount: float
    commission_amount: float
    stripe_fee: float
    net_payout: float
    status: str  # "pending", "processing", "completed"
    stripe_transfer_id: Optional[str] = None
    created_at: datetime
    paid_at: Optional[datetime] = None


# ==================== COMMISSION MODELS ====================

class CommissionRate(BaseModel):
    """Default commission rates by category"""
    category: str
    rate: float  # 0.08 - 0.10 (8-10%)
    stripe_fee: float = 0.029  # 2.9%
    stripe_fixed: float = 0.30  # $0.30


class VendorCommissionOverride(BaseModel):
    """Custom commission rate for specific vendor"""
    store_id: str
    rate: float
    created_at: datetime


# ==================== RETURN MODELS ====================

class ReturnRequest(BaseModel):
    order_id: str
    items: List[Dict[str, Any]]
    reason: str
    customer_email: EmailStr
    status: ReturnStatus
    created_at: datetime
    approved_at: Optional[datetime] = None
    refund_amount: Optional[float] = None


# ==================== API REQUEST/RESPONSE MODELS ====================

class CartItem(BaseModel):
    product_id: str
    quantity: int = 1


class ShippingAddress(BaseModel):
    name: str
    street: str
    city: str
    province: str
    postal_code: str
    country: str = "CA"


class ShippingItem(BaseModel):
    name: str
    weight: float = 2.0  # pounds per sealed product


class CartOptimizationRequest(BaseModel):
    items: List[CartItem]
    shipping_address: ShippingAddress


class OptimizedCartItem(BaseModel):
    product_id: str
    quantity: int
    source: str
    source_name: str
    listing_id: str
    product_price: float
    shipping_cost: float
    total_price: float


class CartOptimizationResponse(BaseModel):
    items: List[OptimizedCartItem]
    total_product_price: float
    total_shipping_cost: float
    total_price: float
    savings: float
    currency: str = "CAD"


class CheckoutRequest(BaseModel):
    items: List[Dict[str, Any]]
    customer_email: EmailStr
    shipping_address: ShippingAddress
    user_id: Optional[str] = None
    save_payment_method: bool = False


class ShippingLabelRequest(BaseModel):
    shop: str
    order_id: str
    shipping_address: ShippingAddress
    items: List[ShippingItem]


class ReturnLabelRequest(BaseModel):
    shop: str
    order_id: str
    customer_address: ShippingAddress
    items: List[ShippingItem]


class PaymentCustomerRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    email: EmailStr


class VendorSubscriptionRequest(BaseModel):
    tier: str = Field(..., pattern="^(basic|growth|pro)$")
    contact_email: EmailStr
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class VendorBillingPortalRequest(BaseModel):
    return_url: Optional[str] = None
