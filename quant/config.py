"""Centralized API key & config loader for the quant pipeline.

All env-var access in this package should go through `config.get()` rather
than calling `os.getenv()` directly. This gives us one place to:

  * Load `.env` exactly once
  * Validate required keys with a useful error message
  * Audit which providers are configured (`python config.py`)
  * Avoid accidentally logging secret values

Secrets are never returned by `__repr__` or printed by `print_status()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_LOADED = False
_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(env_path: str | Path | None = None, override: bool = False) -> bool:
    """Load .env exactly once. Returns True if a file was found and loaded."""
    global _ENV_LOADED
    if _ENV_LOADED and not override:
        return True

    path = Path(env_path) if env_path else _ENV_PATH
    if path.exists():
        load_dotenv(path, override=override)
        _ENV_LOADED = True
        return True

    # Fall back to default search (cwd, parents)
    found = load_dotenv(override=override)
    _ENV_LOADED = True
    return found


def get(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Get an env var. If required=True, raise with a clear error when missing."""
    load_env()
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required env var: {name}. Add it to {_ENV_PATH} "
            f"(see .env.example). Never paste keys directly into source files."
        )
    return value


def get_int(name: str, default: int) -> int:
    raw = get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_bool(name: str, default: bool = False) -> bool:
    raw = get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Provider:
    """Metadata about an external service and the env vars it needs."""
    name: str
    keys: tuple[str, ...]
    docs_url: str
    purpose: str

    def is_configured(self) -> bool:
        load_env()
        return all(os.getenv(k) for k in self.keys)

    def missing_keys(self) -> list[str]:
        load_env()
        return [k for k in self.keys if not os.getenv(k)]


PROVIDERS: dict[str, Provider] = {
    "alpaca": Provider(
        name="Alpaca",
        keys=("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
        docs_url="https://alpaca.markets",
        purpose="Real OHLCV bars + paper trading",
    ),
    "polygon": Provider(
        name="Polygon.io",
        keys=("POLYGON_API_KEY",),
        docs_url="https://polygon.io",
        purpose="Stock/crypto aggregates (free tier: 5 req/min, 2y history)",
    ),
    "newsapi": Provider(
        name="NewsAPI.org",
        keys=("NEWSAPI_KEY",),
        docs_url="https://newsapi.org",
        purpose="Article search for news sentiment (free tier: 100 req/day, dev-only)",
    ),
}


def status() -> dict[str, bool]:
    """Programmatic check: which providers have all required keys set."""
    return {name: p.is_configured() for name, p in PROVIDERS.items()}


def assert_configured(provider_key: str) -> None:
    """Raise if the named provider isn't fully configured."""
    p = PROVIDERS.get(provider_key)
    if p is None:
        raise KeyError(f"Unknown provider: {provider_key}. Known: {list(PROVIDERS)}")
    if not p.is_configured():
        missing = p.missing_keys()
        raise RuntimeError(
            f"{p.name} not configured. Missing env vars: {', '.join(missing)}. "
            f"Get a key at {p.docs_url} and add to {_ENV_PATH}."
        )


def print_status() -> int:
    """CLI: report which providers are configured. No secret values printed."""
    load_env()
    print(f"Env file:  {_ENV_PATH}  ({'exists' if _ENV_PATH.exists() else 'NOT FOUND — copy .env.example to .env'})")
    print()
    print(f"  {'Provider':<14} {'Status':<10} {'Purpose'}")
    print(f"  {'-'*14} {'-'*10} {'-'*60}")
    any_missing = False
    for _, p in PROVIDERS.items():
        if p.is_configured():
            badge = "[OK]"
        else:
            badge = "[--]"
            any_missing = True
        print(f"  {p.name:<14} {badge:<10} {p.purpose}")
        if not p.is_configured():
            print(f"  {'':<14} {'':<10} missing: {', '.join(p.missing_keys())}  ({p.docs_url})")
    print()
    print("Note: values are never printed. To inspect, read .env directly.")
    return 1 if any_missing else 0


if __name__ == "__main__":
    import sys
    sys.exit(print_status())
