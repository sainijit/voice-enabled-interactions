"""Pydantic v2 DTOs for the ordering domain."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float


# ---------------------------------------------------------------------------
# Order items
# ---------------------------------------------------------------------------


class OrderItemIn(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1)


class RemoveOrderItem(BaseModel):
    """A cart line the customer wants taken off their order.

    Distinct from :class:`OrderItemIn` because removal has an extra state that
    addition does not: "remove all of it". ``OrderItemIn.quantity`` is
    constrained to ``>= 1``, so it cannot express that.

    Attributes:
        product_id: Resolved catalogue product id to remove.
        quantity: Units to remove, or ``None`` to remove the entire line
            regardless of how many units it holds.
    """

    product_id: str
    quantity: int | None = Field(default=None, ge=1)


class OrderItem(BaseModel):
    id: int
    order_id: int
    product_id: str
    product_name: str
    quantity: int
    price: float
    subtotal: float


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


OrderStatus = Literal["draft", "confirmed"]


class Order(BaseModel):
    order_id: int
    user_id: str
    status: OrderStatus
    total: float
    created_at: datetime
    items: list[OrderItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request / Response schemas used by REST endpoints
# ---------------------------------------------------------------------------


class CreateOrderRequest(BaseModel):
    user_id: str = "anonymous"
    items: list[OrderItemIn]


class UpdateOrderItemsRequest(BaseModel):
    items: list[OrderItemIn]


class UpsellRequest(BaseModel):
    product_ids: list[str] = Field(description="Products currently in the cart")


# ---------------------------------------------------------------------------
# Upsell
# ---------------------------------------------------------------------------


class UpsellSuggestion(BaseModel):
    product: Product
    reason: str
