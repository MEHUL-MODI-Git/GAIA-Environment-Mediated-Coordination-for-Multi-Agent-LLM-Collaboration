"""Base LLM interface and common types"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..utils import Registry


class ModelTier(str, Enum):
    """Model tier for routing (Feature B)"""

    FAST = "fast"  # Cheap, parallel models (gpt-4o-mini, claude-3-5-haiku, etc.)
    SLOW = "slow"  # Expensive, high-quality models (gpt-4o, claude-3-5-sonnet, etc.)


class LLMResult(BaseModel):
    """Result from an LLM call"""

    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: Optional[str] = None


class BaseLLM(ABC, BaseModel):
    """Base LLM interface with unified message format"""

    model: str
    tier: ModelTier
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0

    @abstractmethod
    async def agenerate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResult:
        """Generate completion from messages

        Args:
            messages: List of message dicts with 'role' and 'content'
                     [{"role": "user", "content": "..."}, ...]

        Returns:
            LLMResult with completion and metadata
        """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name (openai, anthropic, groq, gemini)"""

    @abstractmethod
    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD for token usage"""

    class Config:
        arbitrary_types_allowed = True


# Global registry for LLM providers
llm_registry = Registry(name="LLMRegistry")
