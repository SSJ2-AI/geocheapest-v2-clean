"""
Seed initial products for GeoCheapest v2
7 starter products (3 direct, 4 affiliate)
"""
from database import db
from datetime import datetime


def seed_products():
    """Seed 7 starter products"""
    
    # ==================== DIRECT PRODUCTS (3) ====================
    
    # 1. Pokemon 151
    pokemon_151_ref = db.collection("products").document()
    pokemon_151_ref.set({
        "name": "Pokemon TCG: Scarlet & Violet - 151 Booster Box",
        "description": "Pokemon 151 features all original 151 Pokemon from Kanto region. 36 packs per box.",
        "category": "Pokemon",
        "image_url": "https://product-images.tcgplayer.com/fit-in/874x874/465472.jpg",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_sales": 0
    })
    
    # Add affiliate listing (Amazon)
    db.collection("affiliateProducts").document("amazon_pokemon_151").set({
        "product_id": pokemon_151_ref.id,
        "affiliate_name": "Amazon.ca",
        "affiliate_url": "https://www.amazon.ca/dp/B0C3H1234?tag=geocheapest-20",
        "asin": "B0C3H1234",
        "price": 149.91,
        "in_stock": True,
        "estimated_shipping": 0.0,
        "commission_rate": 0.03,
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # 2. One Piece Two Legends
    op_two_legends_ref = db.collection("products").document()
    op_two_legends_ref.set({
        "name": "One Piece Card Game: Two Legends Booster Box",
        "description": "Two Legends features dual leader cards and powerful new mechanics. 24 packs per box.",
        "category": "One Piece",
        "image_url": "https://product-images.tcgplayer.com/fit-in/874x874/480123.jpg",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_sales": 0
    })
    
    db.collection("affiliateProducts").document("amazon_op_two_legends").set({
        "product_id": op_two_legends_ref.id,
        "affiliate_name": "Amazon.ca",
        "affiliate_url": "https://www.amazon.ca/dp/B0D1F5678?tag=geocheapest-20",
        "asin": "B0D1F5678",
        "price": 199.99,
        "in_stock": True,
        "estimated_shipping": 0.0,
        "commission_rate": 0.03,
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # 3. Pokemon White Flare
    pokemon_white_flare_ref = db.collection("products").document()
    pokemon_white_flare_ref.set({
        "name": "Pokemon TCG: Twilight Masquerade - White Flare Elite Trainer Box",
        "description": "White Flare ETB with exclusive promo card and accessories.",
        "category": "Pokemon",
        "image_url": "https://product-images.tcgplayer.com/fit-in/874x874/490567.jpg",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_sales": 0
    })
    
    db.collection("affiliateProducts").document("amazon_white_flare").set({
        "product_id": pokemon_white_flare_ref.id,
        "affiliate_name": "Amazon.ca",
        "affiliate_url": "https://www.amazon.ca/dp/B0C8K9012?tag=geocheapest-20",
        "asin": "B0C8K9012",
        "price": 146.87,
        "in_stock": True,
        "estimated_shipping": 0.0,
        "commission_rate": 0.03,
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # ==================== AFFILIATE PRODUCTS (4) ====================
    
    # 4. One Piece Legacy
    op_legacy_ref = db.collection("products").document()
    op_legacy_ref.set({
        "name": "One Piece Card Game: Premium Collection Legacy",
        "description": "Premium collection featuring rare legacy cards and exclusive artwork.",
        "category": "One Piece",
        "image_url": "https://product-images.tcgplayer.com/fit-in/874x874/475234.jpg",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_sales": 0
    })
    
    db.collection("affiliateProducts").document("amazon_op_legacy").set({
        "product_id": op_legacy_ref.id,
        "affiliate_name": "Amazon.ca",
        "affiliate_url": "https://www.amazon.ca/dp/B0D2G6789?tag=geocheapest-20",
        "asin": "B0D2G6789",
        "price": 199.99,
        "in_stock": True,
        "estimated_shipping": 0.0,
        "commission_rate": 0.03,
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # 5. Pokemon Black Bolt
    pokemon_black_bolt_ref = db.collection("products").document()
    pokemon_black_bolt_ref.set({
        "name": "Pokemon TCG: Twilight Masquerade - Black Bolt Elite Trainer Box",
        "description": "Black Bolt ETB with exclusive promo card and accessories.",
        "category": "Pokemon",
        "image_url": "https://product-images.tcgplayer.com/fit-in/874x874/490568.jpg",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_sales": 0
    })
    
    db.collection("affiliateProducts").document("cc_black_bolt").set({
        "product_id": pokemon_black_bolt_ref.id,
        "affiliate_name": "Amazon.ca",
        "affiliate_url": "https://www.amazon.ca/dp/B0CX123456?tag=geocheapest-20",
        "price": 120.02,
        "in_stock": True,
        "estimated_shipping": 10.0,
        "commission_rate": 0.05,
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # 6. One Piece Royal Blood
    op_royal_blood_ref = db.collection("products").document()
    op_royal_blood_ref.set({
        "name": "One Piece Card Game: Royal Blood Starter Deck",
        "description": "Ready-to-play starter deck featuring the Royal Blood theme.",
        "category": "One Piece",
        "image_url": "https://product-images.tcgplayer.com/fit-in/874x874/482345.jpg",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_sales": 0
    })
    
    db.collection("affiliateProducts").document("cc_royal_blood").set({
        "product_id": op_royal_blood_ref.id,
        "affiliate_name": "Amazon.ca",
        "affiliate_url": "https://www.amazon.ca/dp/B0CROYALBLOOD?tag=geocheapest-20",
        "price": 88.00,
        "in_stock": True,
        "estimated_shipping": 8.0,
        "commission_rate": 0.05,
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # 7. Pokemon Destined Rivals
    pokemon_destined_rivals_ref = db.collection("products").document()
    pokemon_destined_rivals_ref.set({
        "name": "Pokemon TCG: Destined Rivals Premium Collection",
        "description": "Ultra-premium collection featuring rivalry-themed cards and exclusive promos.",
        "category": "Pokemon",
        "image_url": "https://product-images.tcgplayer.com/fit-in/874x874/495678.jpg",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_sales": 0
    })
    
    db.collection("affiliateProducts").document("cc_destined_rivals").set({
        "product_id": pokemon_destined_rivals_ref.id,
        "affiliate_name": "Amazon.ca",
        "affiliate_url": "https://www.amazon.ca/dp/B0CDRIVALS?tag=geocheapest-20",
        "price": 649.99,
        "in_stock": True,
        "estimated_shipping": 15.0,
        "commission_rate": 0.05,
        "status": "active",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    
    # ==================== DEFAULT COMMISSION RATES ====================
    
    db.collection("commissionRates").document("default").set({
        "Pokemon": 0.10,
        "One Piece": 0.10,
        "Yu-Gi-Oh": 0.10,
        "Magic: The Gathering": 0.10,
        "Dragon Ball": 0.10,
        "Other": 0.08,
        "stripe_fee": 0.029,
        "stripe_fixed": 0.30,
        "updated_at": datetime.utcnow()
    })
    
    print("✅ Successfully seeded 7 starter products!")
    print("   - 3 Direct products (Amazon.ca affiliate)")
    print("   - 4 Affiliate products (Amazon.ca)")


if __name__ == "__main__":
    seed_products()
