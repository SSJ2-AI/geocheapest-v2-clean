from typing import Dict, List, Any
from pydantic import BaseModel

class FilterOption(BaseModel):
    label: str
    value: str

class FilterConfig(BaseModel):
    id: str
    label: str
    type: str # select, range, checkbox
    options: List[FilterOption] = []

class NicheSettings(BaseModel):
    id: str
    name: str
    domain_keywords: List[str]
    filters: List[FilterConfig]
    categories: List[str]

# Configuration for different niches
# This allows the same backend to serve "GeoCheapest TCG", "GeoCheapest Sneakers", etc.
NICHES: Dict[str, NicheSettings] = {
    "tcg": NicheSettings(
        id="tcg",
        name="Trading Card Games",
        domain_keywords=["pokemon", "magic", "yugioh", "lorcana"],
        categories=["Pokemon", "Magic: The Gathering", "Yu-Gi-Oh", "One Piece", "Lorcana"],
        filters=[
            FilterConfig(
                id="condition",
                label="Condition",
                type="select",
                options=[
                    FilterOption(label="Near Mint", value="NM"),
                    FilterOption(label="Lightly Played", value="LP"),
                    FilterOption(label="Sealed", value="SEALED")
                ]
            ),
            FilterConfig(
                id="grading",
                label="Graded",
                type="checkbox",
                options=[
                    FilterOption(label="PSA", value="PSA"),
                    FilterOption(label="BGS", value="BGS"),
                    FilterOption(label="CGC", value="CGC")
                ]
            )
        ]
    ),
    "sneakers": NicheSettings(
        id="sneakers",
        name="Sneakers",
        domain_keywords=["jordan", "yeezy", "nike"],
        categories=["Jordan", "Yeezy", "Nike", "Adidas"],
        filters=[
            FilterConfig(
                id="size",
                label="Size (US)",
                type="select",
                options=[FilterOption(label=str(i), value=str(i)) for i in range(4, 14)]
            )
        ]
    )
}

def get_niche_config(host: str) -> NicheSettings:
    """Determine niche based on hostname or header"""
    # Default to TCG for now
    return NICHES["tcg"]
