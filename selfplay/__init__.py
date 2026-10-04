"""PPO self-play components for Transformer negotiation."""

from .entry import PoolEntry
from .pool import OpponentPool

__all__ = ["PoolEntry", "OpponentPool"]
