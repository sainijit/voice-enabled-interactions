"""Domain configuration loader for the rag-service.

The rag-service is a generic infrastructure component reused across multiple
applications (Smart Kiosk, Healthcare, Education, etc.).  All use-case-specific
identity, intent vocabulary, and reply phrasing is kept *outside* this service
in the consuming application's ``configs/rag-service/`` directory and mounted
into the container at runtime — exactly the same pattern as the audio-analyzer
and text-to-speech services.

At startup this module reads the path pointed to by ``RAG_DOMAIN_CONFIG_PATH``
and loads the three optional YAML files:

* ``agent_profile.yaml``   — agent identity, system prompt, fallback messages,
                              RAG model/collection names
* ``intent_rules.yaml``    — keyword lists used to build intent-detection regexes
* ``reply_templates.yaml`` — spoken response strings and currency symbol

If ``RAG_DOMAIN_CONFIG_PATH`` is not set or any file is missing, safe generic
defaults are returned so the service still starts and responds — callers are
responsible for checking whether ordering/agent features make sense without
domain config.

Usage::

    from agentic import domain_config

    # At module level (regex compilation happens once at import time)
    _CATALOGUE_RE = domain_config.build_catalogue_regex()

    # At response-format time
    currency = domain_config.get_currency_symbol()
    tpl = domain_config.get_reply_template("category_list")
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state — loaded once at import time
# ---------------------------------------------------------------------------

_profile: dict[str, Any] = {}
_intent: dict[str, Any] = {}
_templates: dict[str, Any] = {}
_loaded = False


def _load() -> None:
    """Load all domain config files from RAG_DOMAIN_CONFIG_PATH (once)."""
    global _profile, _intent, _templates, _loaded
    if _loaded:
        return

    config_dir = os.getenv("RAG_DOMAIN_CONFIG_PATH", "").strip()
    if not config_dir:
        logger.info(
            "[domain_config] RAG_DOMAIN_CONFIG_PATH not set — using generic defaults. "
            "Mount configs/rag-service/ and set the env var to enable domain behaviour."
        )
        _loaded = True
        return

    base = Path(config_dir)
    for attr, filename in (
        ("_profile", "agent_profile.yaml"),
        ("_intent", "intent_rules.yaml"),
        ("_templates", "reply_templates.yaml"),
    ):
        path = base / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            globals()[attr] = data
            logger.info("[domain_config] Loaded %s", path)
        else:
            logger.warning("[domain_config] %s not found — using defaults for this section", path)

    _loaded = True


# Trigger load at import time so regex compilation (module-level in
# ordering_agent.py) sees the keywords immediately.
_load()


# ---------------------------------------------------------------------------
# Identity accessors
# ---------------------------------------------------------------------------

def get_agent_name() -> str:
    """Return the agent name for ADK runtime registration."""
    _load()
    return _profile.get("identity", {}).get("agent_name", "rag_agent")


def get_agent_description() -> str:
    """Return the human-readable agent description."""
    _load()
    return _profile.get("identity", {}).get("description", "RAG-powered AI assistant")


def get_system_prompt() -> str:
    """Return the system prompt for the direct RAG query path."""
    _load()
    return _profile.get("identity", {}).get("system_prompt", "")


def get_out_of_scope_fallback() -> str:
    """Return the spoken reply when an out-of-domain question is rejected."""
    _load()
    return _profile.get("identity", {}).get(
        "out_of_scope_fallback",
        "I can only help with topics related to this service. "
        "Is there something else I can help you with?",
    )


def get_order_claim_fallback() -> str:
    """Return the spoken reply when an order-claim cannot be verified."""
    _load()
    return _profile.get("identity", {}).get(
        "order_claim_fallback",
        "Sorry, I could not complete that just now. Please try again.",
    )


def get_rag_model_name() -> str:
    """Return the OpenVINO/OVMS model alias for this domain."""
    _load()
    return _profile.get("rag", {}).get("model_name", "rag-service")


def get_collection_name() -> str:
    """Return the vector-store collection name for this domain."""
    _load()
    return _profile.get("rag", {}).get("collection_name", "rag-collection")


# ---------------------------------------------------------------------------
# Intent keyword accessors
# ---------------------------------------------------------------------------

def _kw_list(key: str) -> list[str]:
    """Return a keyword list from intent_rules.yaml, or empty list."""
    _load()
    return [str(k) for k in (_intent.get(key) or [])]


def get_catalogue_keywords() -> list[str]:
    """Words that signal a menu/item/price question."""
    return _kw_list("catalogue_keywords")


def get_order_action_keywords() -> list[str]:
    """Phrases that signal an order-mutation intent."""
    return _kw_list("order_action_keywords")


def get_domain_keywords() -> list[str]:
    """Broad vocabulary that marks a turn as in-domain."""
    return _kw_list("domain_keywords")


def get_knowledge_keywords() -> list[str]:
    """Outlet/KB query words (hours, address, facilities)."""
    return _kw_list("knowledge_keywords")


def get_category_names() -> tuple[str, ...]:
    """Category name synonyms used to route _force_catalogue() calls."""
    return tuple(_kw_list("category_names"))


# ---------------------------------------------------------------------------
# Regex builders — called once at ordering_agent module load
# ---------------------------------------------------------------------------

def _build_regex(keywords: list[str], flags: int = re.IGNORECASE) -> re.Pattern[str] | None:
    """Build a word-boundary alternation regex from a keyword list.

    Returns ``None`` when the list is empty so callers can skip the guard.
    """
    if not keywords:
        return None
    # Escape special chars, sort longest-first to avoid prefix shadowing
    alts = sorted(
        (re.escape(k) for k in keywords),
        key=len,
        reverse=True,
    )
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", flags)


def build_catalogue_regex() -> re.Pattern[str] | None:
    """Regex matching catalogue (menu/item/price) questions."""
    return _build_regex(get_catalogue_keywords())


def build_domain_regex() -> re.Pattern[str] | None:
    """Regex matching in-domain vocabulary."""
    return _build_regex(get_domain_keywords())


def build_knowledge_regex() -> re.Pattern[str] | None:
    """Regex matching outlet knowledge queries."""
    return _build_regex(get_knowledge_keywords())


# ---------------------------------------------------------------------------
# Reply template accessors
# ---------------------------------------------------------------------------

def get_currency_symbol() -> str:
    """Currency symbol prepended to prices (e.g. '₹' or '$').

    Defaults to '₹' when no domain config is mounted so that existing
    Smart Kiosk deployments that do not yet have the volume mount still
    behave identically to the pre-decoupling code.
    """
    _load()
    return str(_templates.get("currency_symbol", "₹"))


def get_reply_template(name: str) -> str | None:
    """Return a named reply template string, or None if not defined.

    Template placeholders use Python str.format() syntax:
    ``{categories}``, ``{products}``, ``{items}``, ``{total}``, ``{currency}``

    Args:
        name: One of ``category_list``, ``product_list``, ``not_found``,
              ``item_added``, ``order_confirmed``, ``item_removed``.

    Returns:
        The template string, or ``None`` when the key is not present in the
        domain config (caller should fall back to a hardcoded default or the
        generic LLM narration path).
    """
    _load()
    value = _templates.get(name)
    return str(value) if value is not None else None
