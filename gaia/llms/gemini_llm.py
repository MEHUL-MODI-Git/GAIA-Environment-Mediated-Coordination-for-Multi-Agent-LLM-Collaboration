"""Google Gemini LLM provider"""

import time
from typing import List, Dict, Any
import google.generativeai as genai

from .base import BaseLLM, LLMResult, ModelTier, llm_registry


# Gemini pricing per 1M tokens (as of Feb 2024)
GEMINI_PRICING = {
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free tier
}


@llm_registry.register("gemini")
class GeminiLLM(BaseLLM):
    """Google Gemini provider"""

    api_key: str = ""
    _model_instance: Any = None  # genai.GenerativeModel

    def model_post_init(self, __context):
        """Initialize Gemini client after model creation"""
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self._model_instance = genai.GenerativeModel(self.model)

    async def agenerate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResult:
        """Generate completion using Gemini API"""
        start_time = time.time()

        try:
            # Convert messages to Gemini format
            # Gemini uses a different message format: role "user" or "model"
            gemini_messages = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})

            # Generate response
            response = await self._model_instance.generate_content_async(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.temperature),
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                    top_p=kwargs.get("top_p", self.top_p),
                ),
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract content and token counts
            content = response.text if response.text else ""

            # Gemini provides token counts in usage_metadata
            usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
            prompt_tokens = usage.prompt_token_count if usage else 0
            completion_tokens = usage.candidates_token_count if usage else 0

            # Calculate cost
            cost_usd = self.get_cost(prompt_tokens, completion_tokens)

            return LLMResult(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=self.model,
                provider="gemini",
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                finish_reason=response.candidates[0].finish_reason.name
                if response.candidates
                else None,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMResult(
                content=f"Error: {str(e)}",
                model=self.model,
                provider="gemini",
                latency_ms=latency_ms,
                cost_usd=0.0,
            )

    def get_provider_name(self) -> str:
        return "gemini"

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD"""
        if self.model not in GEMINI_PRICING:
            # Default to gemini-1.5-pro pricing if model not found
            pricing = GEMINI_PRICING["gemini-1.5-pro"]
        else:
            pricing = GEMINI_PRICING[self.model]

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    model_config = {"arbitrary_types_allowed": True}


from typing import Any  # Add at top
