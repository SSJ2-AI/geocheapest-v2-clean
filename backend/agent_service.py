import logging
from typing import Dict, Any, List, Optional
from analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

class AgentService:
    """
    AI Agent Service for GeoCheapest.
    Roles:
    1. Catalog Guardian: Normalizes messy vendor data.
    2. Deal Hunter: Chat assistant for users.
    """
    def __init__(self):
        self.analytics = AnalyticsService()
        # In production, initialize OpenAI/Gemini client here
        # self.llm = OpenAI(...)
        pass

    async def normalize_product(self, raw_title: str, raw_description: str) -> Dict[str, Any]:
        """
        Uses LLM to extract canonical product data from messy vendor listings.
        Example: "Pika Base Set 1st Ed" -> {name: "Pikachu", set: "Base Set", edition: "1st", ...}
        """
        try:
            # Mock LLM Logic
            # In reality, this would prompt GPT-4 to extract JSON
            logger.info(f"AGENT: Normalizing '{raw_title}'")
            
            # SAFETY CHECK: If confidence is low, DO NOT return canonical name
            # This forces the system to use the original title and flag for manual review
            confidence = 0.96 # Mock high confidence
            
            if confidence < 0.95:
                logger.warning(f"AGENT: Low confidence ({confidence}) for '{raw_title}'. Flagging for review.")
                return {
                    "canonical_name": None, # Signal to keep original
                    "confidence": confidence,
                    "needs_review": True
                }

            return {
                "canonical_name": raw_title.strip(), # Placeholder
                "confidence": confidence,
                "tags": ["extracted_by_ai"]
            }
        except Exception as e:
            logger.error(f"AGENT: Normalization failed: {e}")
            return {"canonical_name": None, "confidence": 0.0}

    async def standardize_description(self, raw_desc: str) -> str:
        """
        Rewrites vendor descriptions to be standard, professional, and SEO-friendly.
        """
        # Mock Logic
        return f"Standardized: {raw_desc[:50]}... [Full Specs Included]"

    def get_welcome_message(self, user_name: str) -> str:
        """
        Proactive greeting to start the conversation.
        """
        return f"Hi {user_name}! I'm your GeoCheapest Deal Hunter. What are you looking for today? I can help you find competitive decks, investment grade boxes, or specific singles."

    async def chat(self, user_query: str, context: Dict[str, Any], db: Any = None) -> str:
        """
        Shopping Assistant Chatbot with Multi-turn logic.
        """
        logger.info(f"AGENT: User Query: {user_query}")
        
        # 1. Log Query for Analytics (User Intent Tracking)
        await self.analytics.log_event(
            "user_behavior", 
            "chat_queries", 
            {
                "user_id": context.get("user_id"),
                "query": user_query,
                "timestamp": "now"
            }
        )

        query_lower = user_query.lower()
        
        # Mock Multi-turn Logic
        
        # 1. Investment Flow
        if "investment" in query_lower:
            # Infer Preference: User is interested in investing
            logger.info(f"AGENT: Inferring tag 'investor' for user {context.get('user_id')}")
            
            # PERSISTENCE: Save this preference to Firestore
            if db and context.get("user_id"):
                try:
                    user_ref = db.collection("users").document(context["user_id"])
                    # Use array_union to avoid duplicates
                    await user_ref.update({
                        "preferences.intent_tags": firestore.ArrayUnion(["investor"])
                    })
                except Exception as e:
                    logger.error(f"Failed to update user preferences: {e}")
            
            # ACTION-ORIENTED RESPONSE: Don't just analyze, SUGGEST.
            return (
                "Evolving Skies is definitely a strong hold (up 20% recently). "
                "Based on your interest in high-growth sealed product, here are the top 3 investment picks currently in stock:\n\n"
                "1. **Crown Zenith Elite Trainer Box** ($55 CAD) - *Currently Undervalued*\n"
                "2. **Lost Origin Booster Box** ($140 CAD) - *Rising Trend*\n"
                "3. **151 Ultra Premium Collection** ($120 CAD) - *High Long-Term Demand*\n\n"
                "Would you like to add any of these to your cart?"
            )

        # 2. Competitive Play Flow
        if "deck" in query_lower:
            return (
                "The 'Charizard ex' deck is currently dominating the meta. "
                "Here are the best bundles I found for you:\n\n"
                "1. **Charizard ex League Battle Deck** - Best Price: $34.99 (401 Games)\n"
                "2. **Trainers Toolkit (2024)** - Best Price: $29.99 (Hobbiesville)\n\n"
                "Shall I optimize a cart for the full deck list?"
            )

        # 3. General Search
        return "I can help you find the best deals. Tell me what you play or collect, and I'll generate a list of top recommendations for you."
