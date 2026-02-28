"""
Market Data Collector
Responsible for fetching, validating, and storing real-time market data from exchanges.
"""
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import ccxt
from datetime import datetime, timezone
from config import ADMDNNConfig

class MarketDataCollector:
    """Collects market data from various exchanges"""
    
    def __init__(self, config: ADMDNNConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchange_clients: Dict[str, ccxt.Exchange] = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self) -> None:
        """Initialize connections to configured exchanges"""
        for name, exchange_config in self.config.exchanges.items():
            try:
                exchange_class = getattr(ccxt, name)
                exchange_params = {
                    'apiKey': exchange_config.api_key,
                    'secret': exchange_config.api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                }
                
                if exchange_config.sandbox:
                    exchange_params['sandbox'] = True
                    
                exchange = exchange_class(exchange_params)
                exchange.load_markets()
                
                # Test connectivity
                exchange.fetch_time()
                self.exchange_clients[name] = exchange
                self.logger.info(f"Successfully connected to {name} exchange")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {name} exchange: {str(e)}")
                
    async def collect_ohlcv(
        self, 
        exchange_name: str, 
        symbol: str, 
        timeframe: str = '1m',