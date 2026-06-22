"""MetaTrader5 only ships Windows wheels (it talks to the local terminal
over a native API), so it can't be installed in this Linux dev/CI
environment. Stub it out before any `stax.*` module imports it, so the
no-lookahead unit tests (which exercise the pure feature/label logic, not
live MT5 I/O) can run anywhere. Real connectivity is exercised on the
Windows box running the IUX terminal.
"""
import sys
import types


def _install_mt5_stub() -> None:
    if "MetaTrader5" in sys.modules:
        return
    stub = types.ModuleType("MetaTrader5")
    stub.TIMEFRAME_M5 = 5
    stub.TIMEFRAME_M15 = 15
    stub.TIMEFRAME_H1 = 60
    stub.TIMEFRAME_H4 = 240
    stub.ORDER_TYPE_BUY = 0
    stub.ORDER_TYPE_SELL = 1
    stub.TRADE_ACTION_DEAL = 1
    stub.POSITION_TYPE_BUY = 0

    stub.copy_rates_from_pos = lambda *a, **k: None
    stub.symbol_info = lambda *a, **k: None
    stub.symbol_info_tick = lambda *a, **k: None
    stub.symbol_select = lambda *a, **k: True
    stub.initialize = lambda *a, **k: True
    stub.login = lambda *a, **k: True
    stub.shutdown = lambda *a, **k: None
    stub.last_error = lambda: (0, "stub")
    stub.positions_get = lambda *a, **k: []
    stub.order_send = lambda *a, **k: None

    sys.modules["MetaTrader5"] = stub


_install_mt5_stub()
