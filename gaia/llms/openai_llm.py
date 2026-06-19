"""OpenAI LLM provider"""

import time
from typing import List, Dict
import openai
from openai import AsyncOpenAI

from .base import BaseLLM, LLMResult, ModelTier, llm_registry


# Pricing per 1M tokens (as of Feb 2024)
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@llm_registry.register("openai")
class OpenAILLM(BaseLLM):
    """OpenAI LLM provider using SDK v1.x"""

    api_key: str = ""
    client: AsyncOpenAI = None  # type: ignore

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.client:
            object.__setattr__(self, "client", AsyncOpenAI(api_key=self.api_key or None))

    async def agenerate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResult:
        """Generate completion using OpenAI API"""
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
                provider="openai",
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                finish_reason=finish_reason,
            )

        except Exception as e:
            # Return error result
            latency_ms = (time.time() - start_time) * 1000
            return LLMResult(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="openai",
                latency_ms=latency_ms,
                cost_usd=0.0,
            )

    def get_provider_name(self) -> str:
        return "openai"

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD"""
        if self.model not in OPENAI_PRICING:
            # Default to gpt-4o pricing if model not found
            pricing = OPENAI_PRICING["gpt-4o"]
        else:
            pricing = OPENAI_PRICING[self.model]

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

