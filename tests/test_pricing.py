import pytest

from agentic.pricing import cost_usd, model_family, price_for


def test_model_family_returns_opus_for_opus_model_id():
    assert model_family("claude-opus-4-5") == "opus"


def test_model_family_returns_opus_for_versioned_opus_id():
    """Longer versioned IDs are matched by substring."""
    assert model_family("claude-opus-4-8-20260101") == "opus"


def test_model_family_returns_sonnet_for_sonnet_model_id():
    assert model_family("claude-sonnet-4-6") == "sonnet"


def test_model_family_returns_haiku_for_haiku_model_id():
    assert model_family("claude-haiku-4-5-20251001") == "haiku"


def test_model_family_returns_unknown_for_unrecognised_id():
    assert model_family("gpt-4") == "unknown"


def test_model_family_is_case_insensitive():
    """Model IDs from the SDK may arrive with mixed case."""
    assert model_family("Claude-Sonnet-4-6") == "sonnet"


def test_price_for_opus_has_correct_input_price():
    assert price_for("opus")["input"] == 15.00


def test_price_for_opus_has_correct_output_price():
    assert price_for("opus")["output"] == 75.00


def test_price_for_sonnet_has_correct_input_price():
    assert price_for("sonnet")["input"] == 3.00


def test_price_for_haiku_has_correct_input_price():
    assert price_for("haiku")["input"] == 0.80


def test_price_for_returns_all_required_keys():
    """Callers depend on all four token-category keys being present."""
    table = price_for("sonnet")
    assert {"input", "output", "cache_write", "cache_read"} <= table.keys()


def test_price_for_raises_for_unknown_family():
    with pytest.raises(KeyError):
        price_for("unknown")


def test_cache_write_price_is_1_25x_input_price():
    """Cache write tokens are priced at 1.25x the base input rate."""
    p = price_for("opus")
    assert p["cache_write"] == p["input"] * 1.25


def test_cache_read_price_is_0_10x_input_price():
    """Cache read tokens are priced at 10% of the base input rate."""
    p = price_for("opus")
    assert p["cache_read"] == p["input"] * 0.10


def test_cost_usd_input_only():
    """1M opus input tokens at $15/M costs $15.00."""
    assert cost_usd("claude-opus-4-5", input_tokens=1_000_000) == 15.00


def test_cost_usd_output_only():
    """1M opus output tokens at $75/M costs $75.00."""
    assert cost_usd("claude-opus-4-5", output_tokens=1_000_000) == 75.00


def test_cost_usd_combines_input_and_output():
    """Cost sums independent contributions from input and output tokens."""
    # 100k input at $3/M + 50k output at $15/M = $0.30 + $0.75 = $1.05
    result = cost_usd("claude-sonnet-4-6", input_tokens=100_000, output_tokens=50_000)
    assert result == pytest.approx(1.05, abs=1e-9)


def test_cost_usd_zero_tokens_returns_zero():
    assert cost_usd("claude-sonnet-4-6") == 0.0


def test_cost_usd_cache_write_tokens_billed_correctly():
    """1M sonnet cache_write tokens at $3.75/M costs $3.75."""
    result = cost_usd("claude-sonnet-4-6", cache_write_tokens=1_000_000)
    assert result == 3.75


def test_cost_usd_cache_read_tokens_billed_correctly():
    """1M sonnet cache_read tokens at $0.30/M costs $0.30."""
    result = cost_usd("claude-sonnet-4-6", cache_read_tokens=1_000_000)
    assert result == pytest.approx(0.30, abs=1e-9)


def test_cost_usd_cache_write_costs_more_than_same_input_tokens():
    """Cache write tokens cost 25% more than an equal count of input tokens."""
    input_cost = cost_usd("claude-opus-4-5", input_tokens=500_000)
    cache_write_cost = cost_usd("claude-opus-4-5", cache_write_tokens=500_000)
    assert cache_write_cost == pytest.approx(input_cost * 1.25, rel=1e-9)


def test_cost_usd_cache_read_costs_less_than_same_input_tokens():
    """Cache read tokens cost 90% less than an equal count of input tokens."""
    input_cost = cost_usd("claude-opus-4-5", input_tokens=500_000)
    cache_read_cost = cost_usd("claude-opus-4-5", cache_read_tokens=500_000)
    assert cache_read_cost == pytest.approx(input_cost * 0.10, rel=1e-9)


def test_cost_usd_all_token_types_combined():
    """All four token types contribute independently to the total cost."""
    # 1M input ($15) + 1M output ($75) + 1M cache_write ($18.75) + 1M cache_read ($1.50)
    # = $110.25
    result = cost_usd(
        "claude-opus-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    assert result == 110.25


def test_cost_usd_rounds_to_six_decimal_places():
    """Result is rounded to 6 decimal places, not truncated."""
    # 1 haiku input token at $0.80/M = 0.0000008; the 7th decimal (8) rounds the 6th up
    result = cost_usd("claude-haiku-4-5", input_tokens=1)
    assert result == 0.000001


def test_cost_usd_result_never_exceeds_six_decimal_places():
    result = cost_usd("claude-sonnet-4-6", input_tokens=7, output_tokens=3)
    assert result == round(result, 6)


def test_cost_usd_raises_for_unknown_model():
    with pytest.raises(ValueError, match="unknown model family"):
        cost_usd("gpt-4", input_tokens=1_000)
