"""IdentityService — orchestration layer.

Owns the challenge provider and (in later phases) the storage repository, FAISS
index manager, and OpenVINO inference engines.  Routers call this service only;
they never touch SQLite, FAISS, or inference directly.

Phase 2 status: ``challenge`` is fully functional; ``verify`` and ``register``
return structured "pipeline not ready" responses until the inference layer
(Phases 4–6) is wired in.
"""

from __future__ import annotations

import logging

from identity_core.challenge import ChallengeProvider
from identity_core.config import Settings
from identity_core.models import (
    ChallengeResponse,
    RegisterRequest,
    RegisterResponse,
    VerifyRequest,
    VerifyResponse,
)

logger = logging.getLogger(__name__)


class IdentityService:
    """Coordinates biometric enrolment and verification."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._challenge = ChallengeProvider(settings.prompts)
        # Wired in later phases:
        #   self._profiles  : AbstractProfileRepository  (Phase 3)
        #   self._face_index / self._voice_index : FaissIndexManager (Phase 3)
        #   self._face_engine / self._voice_engine : OpenVINO (Phase 4)
        self._inference_ready = False

    # ── Challenge ────────────────────────────────────────────────────────────

    def get_challenge(self) -> ChallengeResponse:
        challenge_id, prompt_text = self._challenge.next_challenge()
        logger.debug("[IDENTITY] Issued challenge %s", challenge_id)
        return ChallengeResponse(challenge_id=challenge_id, prompt_text=prompt_text)

    # ── Verify ───────────────────────────────────────────────────────────────

    async def verify(self, request: VerifyRequest) -> VerifyResponse:
        """Run multimodal verification.  Both modalities are required."""
        if not request.image_base64 or not request.audio_base64:
            return VerifyResponse(
                verified=False,
                reason="Both face (image) and voice (audio) are required.",
            )
        if not self._inference_ready:
            return VerifyResponse(
                verified=False,
                reason="Inference pipeline not yet available (Phase 4–6 pending).",
            )
        # Phases 4–6 will:
        #   1. decode image/audio
        #   2. extract face (256-d) + voice (192-d) embeddings via OpenVINO
        #   3. FAISS search both indices
        #   4. fuse: 0.6*face + 0.4*voice vs combined_threshold
        #   5. load profile from SQLite, return payload
        raise NotImplementedError  # pragma: no cover

    # ── Register (admin / bootstrap) ─────────────────────────────────────────

    async def register(self, request: RegisterRequest) -> RegisterResponse:
        if not self._inference_ready:
            return RegisterResponse(
                user_id=request.user_id,
                registered=False,
                reason="Inference pipeline not yet available (Phase 4–6 pending).",
            )
        raise NotImplementedError  # pragma: no cover
