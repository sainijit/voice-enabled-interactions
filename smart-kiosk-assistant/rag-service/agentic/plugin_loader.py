"""Dynamic agent plugin loader.

Reads `plugin.agent_module` and `plugin.agent_factory` from agent_profile.yaml
(via domain_config) and imports the agent class/factory at runtime.

This makes agent_endpoints.py domain-agnostic: it never imports
`ordering_agent` directly; instead it calls `load_agent_factory()` which
returns the callable that creates/returns the singleton agent.

For Smart Kiosk:
  plugin:
    agent_module: "plugins.kiosk.ordering_agent"
    agent_factory: "get_ordering_agent"

For a healthcare domain:
  plugin:
    agent_module: "plugins.healthcare.appointment_agent"
    agent_factory: "get_appointment_agent"
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from agentic import domain_config

logger = logging.getLogger(__name__)

_FALLBACK_MODULE = "plugins.kiosk.ordering_agent"
_FALLBACK_FACTORY = "get_ordering_agent"


def load_agent_factory() -> Callable[[], Any]:
    """Import and return the agent factory function for the configured domain.

    The factory is a zero-argument callable that returns the singleton agent
    instance (creating it on first call). It must expose an async `chat()` method.

    Returns:
        The agent factory callable.

    Raises:
        ImportError: If the configured module cannot be imported.
        AttributeError: If the factory function is not found in the module.
    """
    domain_config._load()
    profile = domain_config._profile
    plugin_cfg = profile.get("plugin") or {}
    module_path = plugin_cfg.get("agent_module") or _FALLBACK_MODULE
    factory_name = plugin_cfg.get("agent_factory") or _FALLBACK_FACTORY

    logger.info("[PLUGIN-LOADER] Loading agent: %s.%s", module_path, factory_name)
    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name)
    logger.info("[PLUGIN-LOADER] Agent factory loaded: %r", factory)
    return factory
