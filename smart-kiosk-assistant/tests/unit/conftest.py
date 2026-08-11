"""Shared fixtures for unit tests.

``kiosk_core.audio_session`` imports ``sounddevice``, which requires the
PortAudio shared library at import time. PortAudio is only installed inside
the kiosk-core container, so without this stub every unit test module that
touches an audio session fails at *collection* time on a bare host.

Mirrors the mocking already done in ``tests/functional/conftest.py``. None of
these tests exercise a real capture device.
"""
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("sounddevice", MagicMock())
