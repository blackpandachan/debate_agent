"""Rough cost estimate for a debate, from the token usage ADK records on events.

List prices in USD, checked 2026-11-04. Update them when prices change:
  Gemini:  https://ai.google.dev/gemini-api/docs/pricing
  Claude:  https://platform.claude.com/docs/en/about-claude/pricing

What the estimate leaves out:
  * Gemini's free allowance of 5,000 grounded search queries a month. Search
    is billed at list price, so a light month of play costs less than shown.
  * Claude Haiku 5.5's higher rate for prompts over 100K tokens. Debate prompts
    stay far below that.
  * Prompt-cache discounts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# model prefix -> (USD per 1M input tokens, USD per 1M output tokens).
# Output includes thinking tokens, which both providers bill as output.
TOKEN_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.50, 9.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
SEARCH_QUERY_PRICE = 14.00 / 1000  # Grounding with Google Search, per query


def token_price(model: str) -> tuple[float, float] | None:
    for prefix, price in TOKEN_PRICES.items():
        if model.startswith(prefix):
            return price
    return None


class ModelUsage(BaseModel):
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0  # answer + thinking
    usd: float | None = 0.0  # None when the model has no known price


class CostEstimate(BaseModel):
    models: list[ModelUsage]
    search_queries: int
    search_usd: float
    total_usd: float
    unpriced_models: list[str]

    def markdown(self) -> list[str]:
        rows = [
            "",
            "## Estimated cost",
            "",
            "| Model | Calls | Input tokens | Output tokens | USD |",
            "|---|---|---|---|---|",
        ]
        for m in self.models:
            usd = f"${m.usd:.4f}" if m.usd is not None else "n/a"
            rows.append(
                f"| {m.model} | {m.calls} | {m.input_tokens:,} | {m.output_tokens:,} | {usd} |"
            )
        rows.append(
            f"| Google Search grounding | {self.search_queries} queries | | | ${self.search_usd:.4f} |"
        )
        rows.append(f"| **Total** | | | | **${self.total_usd:.4f}** |")
        rows += [
            "",
            "_List prices; ignores Gemini's 5,000 free grounded searches/month and"
            " cache discounts. See debate_agent/pricing.py._",
        ]
        if self.unpriced_models:
            rows.append(f"_No price on file for: {', '.join(self.unpriced_models)}._")
        return rows


def estimate(
    usage_by_model: dict[str, dict[str, int]], search_queries: int
) -> CostEstimate:
    """``usage_by_model``: model -> {"calls", "input_tokens", "output_tokens"}."""
    models: list[ModelUsage] = []
    unpriced: list[str] = []
    total = 0.0
    for model, u in usage_by_model.items():
        price = token_price(model)
        usd = None
        if price is None:
            unpriced.append(model)
        else:
            usd = (u["input_tokens"] * price[0] + u["output_tokens"] * price[1]) / 1e6
            total += usd
        models.append(ModelUsage(model=model, usd=usd, **u))
    search_usd = search_queries * SEARCH_QUERY_PRICE
    return CostEstimate(
        models=models,
        search_queries=search_queries,
        search_usd=search_usd,
        total_usd=round(total + search_usd, 6),
        unpriced_models=unpriced,
    )


def usage_from_events(
    events: list[Any], model_for_node: dict[str, str]
) -> dict[str, dict[str, int]]:
    """Sums token usage per model over a session's events.

    Usage is attributed to the model each node was configured with (what you
    are billed for), not the version string the API echoes back.
    """
    totals: dict[str, dict[str, int]] = {}
    for event in events:
        usage = event.usage_metadata
        if not usage or event.partial:
            continue
        node = ""
        if event.node_info and event.node_info.path:
            node = event.node_info.path.split("/")[-1].split("@")[0]
        model = model_for_node.get(node) or event.model_version or "unknown"
        t = totals.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        t["calls"] += 1
        t["input_tokens"] += usage.prompt_token_count or 0
        t["output_tokens"] += (usage.candidates_token_count or 0) + (
            usage.thoughts_token_count or 0
        )
    return totals
