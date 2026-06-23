"""Timeframe constants shared by both the live MT5 client and offline data
sources (CSV) — kept dependency-free so non-MT5 tooling doesn't need the
Windows-only MetaTrader5 package installed."""

TF_SECONDS = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400}
