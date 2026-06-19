"""Groq LLM provider (OpenAI-compatible API)"""

import time
from typing import List, Dict
from groq import AsyncGroq

from .base import BaseLLM, LLMResult, ModelTier, llm_registry


# Groq pricing (free tier has rate limits)
# Pricing estimates per 1M tokens
GROQ_PRICING = {
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
}


@llm_registry.register("groq")
class GroqLLM(BaseLLM):
    """Groq LLM provider (fast inference)"""

    api_key: str = ""
    client: AsyncGroq = None  # type: ignore

    def model_post_init(self, __context):
        """Initialize Groq client after model creation"""
        if not self.client:
            self.client = AsyncGroq(api_key=self.api_key or None)

    async def agenerate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResult:
        """Generate completion using Groq API (OpenAI-compatible)"""
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage and content
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason

            # Calculate cost
            cost_usd = self.get_cost(prompt_tokens, completion_tokens)

            return LLMResult(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=self.model,
                provider="groq",
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                finish_reason=finish_reason,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMResult(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="groq",
                latency_ms=latency_ms,
                cost_usd=0.0,
            )

    def get_provider_name(self) -> str:
        return "groq"

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD"""
        if self.model not in GROQ_PRICING:
            # Default to llama-3.1-70b pricing if model not found
            pricing = GROQ_PRICING["llama-3.1-70b-versatile"]
        else:
            pricing = GROQ_PRICING[self.model]

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    model_config = {"arbitrary_types_allowed": True}
