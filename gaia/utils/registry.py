"""Registry pattern for component registration (ported from AgentVerse)"""

from typing import Dict, Any
from pydantic import BaseModel


class Registry(BaseModel):
    """Registry for storing and building classes via decorator pattern.

    Example:
        >>> agent_registry = Registry(name="AgentRegistry")
        >>>
        >>> @agent_registry.register("coder")
        >>> class CoderAgent:
        >>>     pass
        >>>
        >>> agent = agent_registry.build("coder", arg1="value")
    """

    name: str
    entries: Dict[str, Any] = {}

    def register(self, key: str):
        """Decorator to register a class with the given key."""

        def decorator(class_builder):
            self.entries[key] = class_builder
            return class_builder

        return decorator

    def build(self, type: str, **kwargs):
        """Build an instance of the registered class with the given type."""
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the '
                f'.register("{type}") method provided in {self.name} registry'
            )
        return self.entries[type](**kwargs)

    def get_all_entries(self):
        """Get all registered entries."""
        return self.entries

    model_config = {"arbitrary_types_allowed": True}
