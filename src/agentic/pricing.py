"""USD pricing tables and cost calculation for Anthropic Claude models."""
from __future__ import annotations

# Per-million token prices in USD
_PRICING: dict[str, dict[str, float]] = {
    "opus": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,  # 1.25x input
        "cache_read": 1.50,    # 0.10x input
    },
    "sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,   # 1.25x input
        "cache_read": 0.30,    # 0.10x input
    },
    "haiku": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,   # 1.25x input
        "cache_read": 0.08,    # 0.10x input
    },
}


def model_family(model_id: str) -> str:
    """Return "opus", "sonnet", "haiku", or "unknown" for the given model ID."""
    lower = model_id.lower()
    for family in ("opus", "sonnet", "haiku"):
        if family in lower:
            return family
    return "unknown"


def price_for(family: str) -> dict[str, float]:
    """Return the pricing table for the given model family.

    Raises KeyError for unrecognised families.
    """
    return _PRICING[family]


def cost_usd(
    model_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Return the cost in USD rounded to 6 decimal places.

    Raises ValueError for model IDs that map to an unknown family.
    """
    family = model_family(model_id)
    if family == "unknown":
        raise ValueError(f"unknown model family for {model_id!r}")
    p = price_for(family)
    raw = (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_write_tokens * p["cache_write"]
        + cache_read_tokens * p["cache_read"]
    ) / 1_000_000
    return round(raw, 6)
