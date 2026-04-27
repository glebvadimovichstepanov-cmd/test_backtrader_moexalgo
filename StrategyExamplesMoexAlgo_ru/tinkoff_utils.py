"""Utility functions for T-Invest API - getting instrument info by ticker/figi/uid."""
import logging
from typing import Optional, Dict, Any
from pandas import DataFrame

from t_tech.invest import Client
from t_tech.invest.services import InstrumentsService
from t_tech.invest.utils import quotation_to_decimal

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_all_instruments_as_df(token: str) -> DataFrame:
    """Получить все доступные инструменты (акции, облигации, ETF, валюты, фьючерсы)
    
    Args:
        token: Токен доступа к T-Invest API
        
    Returns:
        DataFrame с колонками: name, ticker, class_code, figi, uid, type, 
        min_price_increment, scale, lot, trading_status, api_trade_available_flag,
        currency, exchange, buy_available_flag, sell_available_flag, 
        short_enabled_flag, klong, kshort
    """
    with Client(token) as client:
        instruments: InstrumentsService = client.instruments
        tickers = []
        for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
            try:
                for item in getattr(instruments, method)().instruments:
                    tickers.append(
                        {
                            "name": item.name,
                            "ticker": item.ticker,
                            "class_code": item.class_code,
                            "figi": item.figi,
                            "uid": item.uid,
                            "type": method,
                            "min_price_increment": quotation_to_decimal(
                                item.min_price_increment
                            ),
                            "scale": 9 - len(str(item.min_price_increment.nano)) + 1 if item.min_price_increment.nano else 6,
                            "lot": item.lot,
                            "trading_status": str(item.trading_status),
                            "api_trade_available_flag": item.api_trade_available_flag,
                            "currency": item.currency,
                            "exchange": item.exchange,
                            "buy_available_flag": item.buy_available_flag,
                            "sell_available_flag": item.sell_available_flag,
                            "short_enabled_flag": item.short_enabled_flag,
                            "klong": quotation_to_decimal(item.klong),
                            "kshort": quotation_to_decimal(item.kshort),
                        }
                    )
            except Exception as e:
                logger.warning(f"Ошибка при получении инструментов типа {method}: {e}")
        
        return DataFrame(tickers)


def get_figi_by_ticker(token: str, ticker: str) -> Optional[str]:
    """Получить FIGI по тикуру
    
    Args:
        token: Токен доступа к T-Invest API
        ticker: Тикер инструмента (например, 'VTBR', 'SNGS')
        
    Returns:
        FIGI в формате BBG... или None если не найден
    """
    tickers_df = get_all_instruments_as_df(token)
    ticker_df = tickers_df[tickers_df["ticker"] == ticker]
    if ticker_df.empty:
        logger.error("There is no such ticker: %s", ticker)
        return None
    figi = ticker_df["figi"].iloc[0]
    return figi


def get_ticker_by_uid(token: str, uid: str) -> Optional[Dict[str, Any]]:
    """Получить информацию об инструменте по UID
    
    Args:
        token: Токен доступа к T-Invest API
        uid: UID инструмента
        
    Returns:
        Dict с информацией о тикере (ticker, figi, name, type) или None если не найден
    """
    tickers_df = get_all_instruments_as_df(token)
    uid_df = tickers_df[tickers_df["uid"] == uid]
    if uid_df.empty:
        logger.debug("There is no such uid: %s", uid)
        return None
    row = uid_df.iloc[0]
    return {
        "ticker": row["ticker"],
        "figi": row["figi"],
        "name": row["name"],
        "type": row["type"],
        "currency": row["currency"],
        "exchange": row["exchange"],
    }


def get_ticker_by_figi(token: str, figi: str) -> Optional[Dict[str, Any]]:
    """Получить информацию об инструменте по FIGI
    
    Args:
        token: Токен доступа к T-Invest API
        figi: FIGI инструмента
        
    Returns:
        Dict с информацией о тикере (ticker, name, type, uid) или None если не найден
    """
    tickers_df = get_all_instruments_as_df(token)
    figi_df = tickers_df[tickers_df["figi"] == figi]
    if figi_df.empty:
        logger.debug("There is no such figi: %s", figi)
        return None
    row = figi_df.iloc[0]
    return {
        "ticker": row["ticker"],
        "name": row["name"],
        "type": row["type"],
        "uid": row["uid"],
        "currency": row["currency"],
        "exchange": row["exchange"],
    }


# Кэшированная версия для множественных запросов
class InstrumentCache:
    """Кэш для инструментов чтобы не запрашивать каждый раз все инструменты"""
    
    def __init__(self, token: str):
        self.token = token
        self._df: Optional[DataFrame] = None
        self._uid_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._figi_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._ticker_map: Optional[Dict[str, Dict[str, Any]]] = None
    
    def _ensure_loaded(self):
        """Загрузить данные если еще не загружены"""
        if self._df is None:
            self._df = get_all_instruments_as_df(self.token)
            # Строим словари для быстрого поиска
            self._uid_map = {}
            self._figi_map = {}
            self._ticker_map = {}
            for _, row in self._df.iterrows():
                uid = row["uid"]
                figi = row["figi"]
                ticker = row["ticker"]
                
                if uid:
                    self._uid_map[uid] = {
                        "ticker": ticker,
                        "figi": figi,
                        "name": row["name"],
                        "type": row["type"],
                        "currency": row["currency"],
                        "exchange": row["exchange"],
                    }
                if figi:
                    self._figi_map[figi] = {
                        "ticker": ticker,
                        "name": row["name"],
                        "type": row["type"],
                        "uid": uid,
                        "currency": row["currency"],
                        "exchange": row["exchange"],
                    }
                if ticker:
                    self._ticker_map[ticker] = {
                        "figi": figi,
                        "name": row["name"],
                        "type": row["type"],
                        "uid": uid,
                        "currency": row["currency"],
                        "exchange": row["exchange"],
                    }
    
    def get_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """Получить информацию по UID из кэша"""
        self._ensure_loaded()
        return self._uid_map.get(uid)
    
    def get_by_figi(self, figi: str) -> Optional[Dict[str, Any]]:
        """Получить информацию по FIGI из кэша"""
        self._ensure_loaded()
        return self._figi_map.get(figi)
    
    def get_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Получить информацию по тикуру из кэша"""
        self._ensure_loaded()
        return self._ticker_map.get(ticker)
    
    def get_figi_by_ticker(self, ticker: str) -> Optional[str]:
        """Получить FIGI по тикуру из кэша"""
        info = self.get_by_ticker(ticker)
        if info:
            return info["figi"]
        return None
