"""Trade cost model — deliberately harsh, per the architecture doc's
warning that cost eats the entire edge if you go easy here."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostParams:
    spread_pts: float          # typical spread in price points
    point: float                # MT5 symbol point size
    commission_per_lot: float = 0.0
    slippage_pts: float = 1.0


def trade_cost(cost: CostParams) -> float:
    """Round-trip cost in price units for a 1-lot-equivalent trade."""
    spread_cost = cost.spread_pts * cost.point
    slippage_cost = cost.slippage_pts * cost.point
    return spread_cost + slippage_cost  # commission applied separately in $ terms


def pnl_net(direction: int, open_px: float, close_px: float, cost: CostParams, lots: float = 1.0) -> float:
    """direction: +1 long, -1 short. Returns net PnL in price units (not $)."""
    gross = direction * (close_px - open_px)
    cost_px = trade_cost(cost)
    return gross - cost_px
