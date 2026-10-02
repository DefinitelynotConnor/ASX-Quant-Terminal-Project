"""
src/data
========
Data ingestion and cleaning layer for the ASX Quant Terminal.

Modules
-------
asx_data_loader  : Fetch and cache ASX equity OHLCV price data
macro_data_loader: Fetch macroeconomic variables (rates, VIX, FX, commodities)
data_cleaning    : Price and return cleaning, normalisation, train/test split
"""

from .asx_data_loader import ASXDataLoader, fetch_price_data, fetch_universe_data
from .macro_data_loader import MacroDataLoader
from .data_cleaning import DataCleaner

__all__ = [
    "ASXDataLoader",
    "MacroDataLoader",
    "DataCleaner",
    "fetch_price_data",
    "fetch_universe_data",
]
