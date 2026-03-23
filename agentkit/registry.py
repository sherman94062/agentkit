"""Tool registry — register tools and expose them to agents."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from .exceptions import ToolError

# Python type → JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_from_hints(fn: Callable) -> dict:
    """Auto-build a JSON Schema input_schema from function type hints."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        hint = hints.get(name, str)
        json_type = _TYPE_MAP.get(hint, "string")
        prop: dict[str, Any] = {"type": json_type}

        # Use docstring-style description if available, else just the name
        prop["description"] = name.replace("_", " ")
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str
    input_schema: dict
    output_schema: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class ToolRegistry:
    """Register tools and expose them to agents."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        tags: list[str] | None = None,
        input_schema: dict | None = None,
    ):
        """Decorator: @registry.register('my_tool', 'Does X')"""

        def decorator(fn: Callable) -> Callable:
            schema = input_schema or _schema_from_hints(fn)
            spec = ToolSpec(
                name=name,
                fn=fn,
                description=description,
                input_schema=schema,
                tags=tags or [],
            )
            self._tools[name] = spec
            return fn

        return decorator

    def add(
        self,
        name: str,
        fn: Callable,
        description: str,
        tags: list[str] | None = None,
        input_schema: dict | None = None,
    ) -> None:
        """Imperative registration (non-decorator)."""
        schema = input_schema or _schema_from_hints(fn)
        self._tools[name] = ToolSpec(
            name=name,
            fn=fn,
            description=description,
            input_schema=schema,
            tags=tags or [],
        )

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolError(name, "Tool not found in registry")
        return self._tools[name]

    def list(self, tags: list[str] | None = None) -> list[ToolSpec]:
        if not tags:
            return list(self._tools.values())
        return [
            t for t in self._tools.values() if any(tag in t.tags for tag in tags)
        ]

    def to_anthropic_tools(self, names: list[str] | None = None) -> list[dict]:
        """Return tool specs in Anthropic API format."""
        if names:
            specs = [self.get(n) for n in names]
        else:
            specs = list(self._tools.values())
        return [
            {
                "name": s.name,
                "description": s.description,
                "input_schema": s.input_schema,
            }
            for s in specs
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
