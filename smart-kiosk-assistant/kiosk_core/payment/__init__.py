"""Demo payment feature for the kiosk checkout flow.

Generates a scannable payment QR code for a *confirmed* order so the customer
kiosk screen can show it under the cart.

⚠️  DEMO ONLY — the encoded payee handle is deliberately non-routable, so no
real money can ever move. See ``configs/ordering/payment.yaml``.
"""

from kiosk_core.payment.models import PaymentIntent
from kiosk_core.payment.service import PaymentService, load_payment_config

__all__ = ["PaymentIntent", "PaymentService", "load_payment_config"]
