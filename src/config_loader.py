"""Configuration loader: reads YAML config files and returns model objects."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models import ProductInfo, Dimension

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(path: str | None = None) -> dict[str, Any]:
    """Load system settings from settings.yaml."""
    p = Path(path) if path else _CONFIG_DIR / "settings.yaml"
    return _load_yaml(p)


def load_dimensions(path: str | None = None) -> list[Dimension]:
    """Load dimensions from dimensions.yaml."""
    p = Path(path) if path else _CONFIG_DIR / "dimensions.yaml"
    data = _load_yaml(p)
    return [Dimension(**d) for d in data.get("dimensions", [])]


def load_products(path: str | None = None) -> list[ProductInfo]:
    """Load products from products.yaml."""
    p = Path(path) if path else _CONFIG_DIR / "products.yaml"
    data = _load_yaml(p)
    return [ProductInfo(**prod) for prod in data.get("products", [])]


def save_dimensions(dimensions: list[Dimension], path: str | None = None) -> None:
    """Persist dimensions back to YAML."""
    p = Path(path) if path else _CONFIG_DIR / "dimensions.yaml"
    data = {"dimensions": [d.model_dump() for d in dimensions]}
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def save_products(products: list[ProductInfo], path: str | None = None) -> None:
    """Persist products back to YAML."""
    p = Path(path) if path else _CONFIG_DIR / "products.yaml"
    data = {"products": [p_.model_dump() for p_ in products]}
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def save_settings(settings: dict[str, Any], path: str | None = None) -> None:
    """Persist settings back to YAML."""
    p = Path(path) if path else _CONFIG_DIR / "settings.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(settings, f, allow_unicode=True, sort_keys=False)
