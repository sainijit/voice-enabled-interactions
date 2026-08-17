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

import json
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
_guard_rules: dict[str, Any] | None = None
_loaded = False


def _config_base_path() -> Path | None:
    """Return the mounted domain-config directory, if configured."""
    raw = os.getenv("RAG_DOMAIN_CONFIG_PATH", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path.parent if path.suffix else path


def _load() -> None:
    """Load all domain config files from RAG_DOMAIN_CONFIG_PATH (once)."""
    global _profile, _intent, _templates, _loaded
    if _loaded:
        return

    base = _config_base_path()
    if base is None:
        logger.info(
            "[domain_config] RAG_DOMAIN_CONFIG_PATH not set — using generic defaults. "
            "Mount configs/rag-service/ and set the env var to enable domain behaviour."
        )
        _loaded = True
        return

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


def _load_guard_rules() -> dict[str, Any]:
    """Load guard_rules.json lazily from the mounted config directory."""
    global _guard_rules
    if _guard_rules is not None:
        return _guard_rules

    base = _config_base_path()
    if base is None:
        _guard_rules = {}
        return _guard_rules

    path = base / "guard_rules.json"
    if not path.exists():
        _guard_rules = {}
        return _guard_rules

    try:
        _guard_rules = json.loads(path.read_text(encoding="utf-8")) or {}
        logger.info("[domain_config] Loaded %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[domain_config] Failed to load %s — using defaults: %s", path, exc)
        _guard_rules = {}
    return _guard_rules


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


# ---------------------------------------------------------------------------
# Guard rule accessors
# ---------------------------------------------------------------------------

def get_guard_rules(guard_name: str) -> dict:
    """Return the guard rules dict for guard_name (e.g. 'menu_guard').
    Returns {} if guard_rules.json is not mounted or guard_name not found.
    """
    rules = _load_guard_rules()
    guard_rules = rules.get(guard_name, {}) if isinstance(rules, dict) else {}
    return guard_rules if isinstance(guard_rules, dict) else {}


def get_guard_rule(guard_name: str, key: str, default: Any = None) -> Any:
    """Get a single value from guard_rules[guard_name][key], with fallback."""
    return get_guard_rules(guard_name).get(key, default)


def get_guard_patterns(guard_name: str) -> list[str]:
    """Return claim_patterns list for guard_name. Returns [] if not found."""
    patterns = get_guard_rule(guard_name, "claim_patterns", [])
    if not isinstance(patterns, list):
        return []
    return [str(pattern) for pattern in patterns]


# ---------------------------------------------------------------------------
# Tool role accessors (Phase 1 decoupling)
# ---------------------------------------------------------------------------
# The rag-service never hardcodes MCP tool names. Tool names are discovered
# at bootstrap from the MCP server (mcp_client.bootstrap_mcp_tools). The
# domain application only defines REGEX PATTERNS in agent_profile.yaml under
# `tool_roles` that classify discovered names into semantic roles.
#
# This eliminates duplication between agent_profile.yaml and the MCP server
# configuration — tool names live in exactly one place: the MCP server.

# Fallback role patterns preserving Smart Kiosk semantics when no domain
# config is mounted — ensures zero behaviour change for existing deployments.
_FALLBACK_ROLE_PATTERNS: dict[str, str] = {
    "cart_mutate":      r"^(place|update)_order$",
    "cart_remove":      r"^(remove_from|cancel)_order$",
    "cart_confirm":     r"^confirm",
    "cart_read":        r"^get_(order|current_order)$",
    "catalogue_read":   r"^(list_|get_popular|get_product)",
    "upsell_read":      r"^get_upsell",
    "user_id_injected": r"^(place|remove_from|cancel)_order$|^confirm_active_order$|^get_current_order$",
    "dietary_injected": r"^(place|update)_order$|^(list_products|get_popular_products)$",
}


def get_tool_role_patterns() -> dict[str, str]:
    """Return role → regex pattern mapping for classifying discovered MCP tools.

    Patterns come from ``tool_roles`` in agent_profile.yaml.  Falls back to
    Smart Kiosk defaults when no domain config is mounted.  Empty-string
    patterns disable a role (e.g. ``dietary_injected: ""`` for healthcare).
    """
    _load()
    raw = _profile.get("tool_roles") or {}
    if not raw:
        return _FALLBACK_ROLE_PATTERNS
    merged = dict(_FALLBACK_ROLE_PATTERNS)
    for key, val in raw.items():
        merged[key] = str(val) if val is not None else ""
    return merged


def classify_discovered_tools(tool_names: list[str]) -> dict[str, frozenset[str]]:
    """Classify a list of MCP-discovered tool names into semantic role buckets.

    Called once after MCP bootstrap with the full list of tool names returned
    by the server.  Returns a dict of role → frozenset so callers can update
    ``action_result.CLAIM_TOOLS`` and the injection sets in ordering_agent.

    Args:
        tool_names: All tool names discovered from the MCP server(s).

    Returns:
        Dict mapping each role key (e.g. ``cart_mutate``) to the frozenset
        of tool names whose name matches that role's pattern.  A role with
        an empty or None pattern matches nothing (empty frozenset).
    """
    patterns = get_tool_role_patterns()
    result: dict[str, set[str]] = {role: set() for role in patterns}

    for name in tool_names:
        for role, pattern in patterns.items():
            if pattern and re.search(pattern, name):
                result[role].add(name)

    classified = {role: frozenset(names) for role, names in result.items()}
    logger.info(
        "[domain_config] Tool classification from %d discovered tools: %s",
        len(tool_names),
        {k: sorted(v) for k, v in classified.items() if v},
    )
    return classified


def get_all_tool_names_from_classification(classified: dict[str, frozenset[str]]) -> tuple[str, ...]:
    """Deduplicate all tool names across roles for ``_TOOL_MENTION_RE`` etc.

    ``knowledge_lookup`` is prepended — it is a rag-service internal tool
    never appearing in the MCP server's tool list.
    """
    seen: set[str] = set()
    names: list[str] = ["knowledge_lookup"]
    for tools in classified.values():
        for t in tools:
            if t not in seen:
                seen.add(t)
                names.append(t)
    return tuple(names)


# ---------------------------------------------------------------------------
# Agent instruction accessors (Phase 2 decoupling)
# ---------------------------------------------------------------------------
# The agent instruction is assembled from three sources:
#   1. domain sections (this file): persona, tool_rules, domain_tags
#   2. core framework (code):       multi-action turns, knowledge block,
#                                   tool errors, output format
#
# Each section is optional — if absent the caller uses its own fallback.

def get_agent_instruction_section(section: str) -> str:
    """Return one named section of the agent instruction from domain config.

    Args:
        section: One of ``persona``, ``tool_rules``, ``domain_tags``.

    Returns:
        The section string stripped of leading/trailing whitespace, or an
        empty string when the section is absent.
    """
    _load()
    raw = _profile.get("agent_instruction", {})
    if not isinstance(raw, dict):
        return ""
    value = raw.get(section, "")
    return str(value).strip() if value else ""


def get_agent_instruction(core_framework: str) -> str:
    """Assemble the full agent instruction for the LLM.

    Combines domain-owned sections (persona, tool_rules, domain_tags) with
    the generic core_framework string provided by ordering_agent.py.  The
    generic section is never domain-specific and is always present; the domain
    sections can be empty (returns core_framework only).

    Args:
        core_framework: The generic, always-present instruction block baked
            into the service code (multi-action turns, knowledge block usage,
            tool error handling, output format constraints).

    Returns:
        The assembled instruction string, ready to pass to LlmAgent.
    """
    persona = get_agent_instruction_section("persona")
    tool_rules = get_agent_instruction_section("tool_rules")
    domain_tags = get_agent_instruction_section("domain_tags")
    parts = [p for p in (persona, tool_rules, core_framework, domain_tags) if p]
    return "\n\n".join(parts).strip()
