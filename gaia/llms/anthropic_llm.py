"""Anthropic Claude LLM provider"""

import time
from typing import List, Dict
import anthropic
from anthropic import AsyncAnthropic

from .base import BaseLLM, LLMResult, ModelTier, llm_registry


# Pricing per 1M tokens (as of Feb 2024)
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}


@llm_registry.register("anthropic")
class AnthropicLLM(BaseLLM):
    """Anthropic Claude provider"""

    api_key: str = ""
    client: AsyncAnthropic = None  # type: ignore

    def model_post_init(self, __context):
        """Initialize Anthropic client after model creation"""
        if not self.client:
            self.client = AsyncAnthropic(api_key=self.api_key or None)

    async def agenerate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResult:
        """Generate completion using Anthropic API"""
        start_time = time.time()

        try:
            # Anthropic requires system messages to be separate
            system_message = ""
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

            response = await self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,  # type: ignore
                system=system_message if system_message else anthropic.NOT_GIVEN,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage and content
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            content = response.content[0].text if response.content else ""
            finish_reason = response.stop_reason

            # Calculate cost
            cost_usd = self.get_cost(prompt_tokens, completion_tokens)

            return LLMResult(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=self.model,
                provider="anthropic",
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                finish_reason=finish_reason,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMResult(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="anthropic",
                latency_ms=latency_ms,
                cost_usd=0.0,
            )

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD"""
        if self.model not in ANTHROPIC_PRICING:
            # Default to sonnet pricing if model not found
            pricing = ANTHROPIC_PRICING["claude-3-5-sonnet-20241022"]
        else:
            pricing = ANTHROPIC_PRICING[self.model]

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    model_config = {"arbitrary_types_allowed": True}
