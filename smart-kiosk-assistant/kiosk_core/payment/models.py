"""Pydantic v2 DTOs for the demo payment domain."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PaymentIntent(BaseModel):
    """Everything the customer screen needs to render a payment QR code.

    Derived on demand from a confirmed :class:`~kiosk_core.ordering.models.Order`
    — nothing is persisted, so there is no payment state that can drift out of
    sync with the order it belongs to.

    Attributes:
        order_id: The confirmed order this intent settles.
        order_ref: Human-readable order reference (e.g. ``"ORD-11"``), matching
            the id the voice agent speaks and the cart header shows.
        amount: Amount payable, copied from the confirmed order total.
        currency: ISO-4217 currency code (e.g. ``"INR"``).
        payee_name: Merchant display name shown under the QR code.
        payload: The raw string encoded inside the QR code (a UPI-style
            ``upi://pay?...`` deep link).
        qr_svg_data_uri: A ``data:image/svg+xml;base64,...`` URI that can be
            dropped straight into an ``<img src>``. SVG (not PNG) so the code
            stays crisp at any kiosk display density, and base64 (not raw
            markup) so the UI never has to inject untrusted HTML.
        qr_size_px: Suggested on-screen render size in pixels (square).
        is_demo: Always ``True`` in this build. Present so the UI renders the
            demo banner from data rather than hardcoding the assumption, and so
            a future real-payment integration only has to flip this field.
        demo_banner: Warning text the UI must display alongside the QR.
        simulated_success_after_seconds: How long the UI shows the QR before
            flipping to a simulated "payment received" confirmation. ``0``
            disables the simulation and leaves the QR up indefinitely — which
            is what a real PSP integration would use, since success would then
            arrive via webhook instead of a timer.
        success_message: Text shown once the simulated payment "succeeds".
    """

    order_id: int
    order_ref: str
    amount: float
    currency: str
    payee_name: str
    payload: str
    qr_svg_data_uri: str
    qr_size_px: int = Field(default=190, ge=64, le=1024)
    is_demo: bool = True
    demo_banner: str
    simulated_success_after_seconds: float = Field(default=3.0, ge=0)
    success_message: str = "Payment received"
