"""Demo payment QR generation.

Builds a UPI-style deep link for a confirmed order and renders it as an SVG
QR code. Pure computation — no SQL, no HTTP, no framework types — so it can be
unit-tested standalone and reused by any caller that already has an ``Order``.

⚠️  DEMO ONLY. The payee handle comes from ``configs/ordering/payment.yaml``
and is a non-routable placeholder, so scanning the QR with a real payment app
cannot transfer money.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import segno
import yaml

from kiosk_core.ordering.models import Order
from kiosk_core.payment.models import PaymentIntent

logger = logging.getLogger(__name__)


# Fallbacks used only when payment.yaml is missing or a key is absent, so a
# misplaced config file degrades to a still-safe demo QR instead of crashing
# the checkout screen. The VPA stays non-routable here too.
_DEFAULTS: dict[str, object] = {
    "payee_name": "Kiosk Demo",
    "payee_vpa": "demo@kiosk.invalid",
    "currency": "INR",
    "note_template": "Kiosk order {order_ref}",
    "demo_banner": "DEMO ONLY — NOT A REAL PAYMENT",
    "qr_size_px": 190,
    "simulated_success_after_seconds": 3,
    "success_message": "Payment received",
}


@dataclass(frozen=True)
class PaymentConfig:
    """Merchant identity and QR payload settings loaded from YAML."""

    payee_name: str
    payee_vpa: str
    currency: str
    note_template: str
    demo_banner: str
    qr_size_px: int
    simulated_success_after_seconds: float
    success_message: str


def load_payment_config(config_path: str) -> PaymentConfig:
    """Load demo payment settings from ``payment.yaml``.

    Args:
        config_path: Path to the YAML file, relative to the project root or
            absolute.

    Returns:
        A populated :class:`PaymentConfig`. Missing file or missing keys fall
        back to safe non-routable defaults rather than raising, so the kiosk
        never fails to boot over a demo-only config.
    """
    raw: dict[str, object] = {}
    path = Path(config_path)
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            raw = (loaded.get("payment") or {}) if isinstance(loaded, dict) else {}
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("[PAYMENT] Could not read %s (%s) — using defaults", config_path, exc)
    else:
        logger.warning("[PAYMENT] Config %s not found — using defaults", config_path)

    merged = {**_DEFAULTS, **raw}
    return PaymentConfig(
        payee_name=str(merged["payee_name"]),
        payee_vpa=str(merged["payee_vpa"]),
        currency=str(merged["currency"]),
        note_template=str(merged["note_template"]),
        demo_banner=str(merged["demo_banner"]),
        qr_size_px=int(merged["qr_size_px"]),  # type: ignore[arg-type]
        simulated_success_after_seconds=max(0.0, float(merged["simulated_success_after_seconds"])),  # type: ignore[arg-type]
        success_message=str(merged["success_message"]),
    )


def format_order_ref(order_id: int) -> str:
    """Format an order id the way the customer sees it (``"ORD-11"``).

    Matches ``formatOrderId`` in the kiosk UI and the id the voice agent
    speaks, so the reference printed under the QR is the same one the customer
    was just told — no zero padding.
    """
    return f"ORD-{order_id}"


class PaymentService:
    """Builds :class:`PaymentIntent` objects for confirmed orders.

    Stateless after construction and safe to share as a singleton.

    Args:
        config: Merchant identity and QR settings.
    """

    def __init__(self, config: PaymentConfig) -> None:
        self._config = config

    @property
    def config(self) -> PaymentConfig:
        """The loaded demo payment configuration."""
        return self._config

    def build_payload(self, order_ref: str, amount: float) -> str:
        """Build the UPI-style deep link encoded in the QR code.

        Args:
            order_ref: Human-readable order reference, e.g. ``"ORD-11"``.
            amount: Amount payable.

        Returns:
            A ``upi://pay?...`` URI. Every value is percent-encoded so an
            ampersand or space in the merchant name cannot corrupt the query
            string.
        """
        cfg = self._config
        note = cfg.note_template.format(order_ref=order_ref)
        params = {
            "pa": cfg.payee_vpa,
            "pn": cfg.payee_name,
            "am": f"{amount:.2f}",
            "cu": cfg.currency,
            "tn": note,
            "tr": order_ref,
        }
        query = "&".join(f"{key}={quote(str(value), safe='')}" for key, value in params.items())
        return f"upi://pay?{query}"

    def render_qr_data_uri(self, payload: str) -> str:
        """Render ``payload`` as a base64 ``data:`` URI holding an SVG QR code.

        SVG keeps the code sharp at any kiosk display density, and base64
        wrapping means the UI can use a plain ``<img src>`` instead of
        injecting raw markup into the DOM.

        Args:
            payload: The string to encode.

        Returns:
            A ``data:image/svg+xml;base64,...`` URI.
        """
        # error='m' (~15% recovery) tolerates glare and smudges on a kiosk
        # touchscreen; the payload is short enough that the extra redundancy
        # costs no meaningful module density.
        qr = segno.make(payload, error="m")

        # Must be the standalone SVG writer, NOT segno's `svg_inline()`:
        # svg_inline omits the xmlns declaration (it relies on the surrounding
        # HTML parser to supply the SVG namespace). The UI loads this through
        # an <img src> instead, where the SVG is parsed as its own document —
        # and a namespace-less <svg> root renders as a blank image there.
        buffer = io.BytesIO()
        qr.save(
            buffer,
            kind="svg",
            scale=10,
            border=2,
            dark="#000000",
            light="#ffffff",
            xmldecl=False,
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"

    def create_intent(self, order: Order) -> PaymentIntent:
        """Build the payment intent for a confirmed order.

        Args:
            order: The order to collect payment for. Must be confirmed —
                showing a payment QR for a cart the customer is still editing
                would quote an amount that changes under them.

        Returns:
            A fully populated :class:`PaymentIntent`.

        Raises:
            ValueError: If ``order`` is not in ``confirmed`` status.
        """
        if order.status != "confirmed":
            raise ValueError(
                f"Order {order.order_id} is {order.status}; payment QR is only "
                "available for a confirmed order."
            )

        order_ref = format_order_ref(order.order_id)
        payload = self.build_payload(order_ref, order.total)
        cfg = self._config

        logger.info(
            "[PAYMENT] Built demo payment intent for %s amount=%.2f %s",
            order_ref, order.total, cfg.currency,
        )
        return PaymentIntent(
            order_id=order.order_id,
            order_ref=order_ref,
            amount=order.total,
            currency=cfg.currency,
            payee_name=cfg.payee_name,
            payload=payload,
            qr_svg_data_uri=self.render_qr_data_uri(payload),
            qr_size_px=cfg.qr_size_px,
            is_demo=True,
            demo_banner=cfg.demo_banner,
            simulated_success_after_seconds=cfg.simulated_success_after_seconds,
            success_message=cfg.success_message,
        )
