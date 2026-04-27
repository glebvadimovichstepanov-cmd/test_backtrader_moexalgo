"""Utility functions for T-Invest API - getting instrument info by ticker/figi/uid."""
import logging
from typing import Optional, Dict, Any

from t_tech.invest import Client

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_instrument_by_ticker(token: str, ticker: str) -> Optional[Dict[str, Any]]:
    """Получить информацию об инструменте по тикуру через find_instrument
    
    Args:
        token: Токен доступа к T-Invest API
        ticker: Тикер инструмента (например, 'VTBR', 'SNGS') - точное совпадение
        
    Returns:
        Dict с информацией о тикере (ticker, figi, name, uid, type) или None если не найден
    """
    try:
        with Client(token) as client:
            # Используем instruments.find_instrument для поиска инструмента по тикуру
            instruments_response = client.instruments.find_instrument(query=ticker)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
                # Ищем инструмент с точным совпадением тикера
                for instrument in instruments_response.instruments:
                    inst_ticker = getattr(instrument, 'ticker', None)
                    # Строгое сравнение тикеров - должно полностью совпадать
                    if inst_ticker == ticker:
                        figi = getattr(instrument, 'figi', None)
                        # Возвращаем только FIGI в формате BBG... (требуется для post_order)
                        if figi and figi.startswith('BBG'):
                            return {
                                "ticker": inst_ticker,
                                "figi": figi,
                                "name": getattr(instrument, 'name', 'N/A'),
                                "uid": getattr(instrument, 'uid', None),
                                "type": getattr(instrument, 'instrument_kind', 'N/A'),
                                "currency": getattr(instrument, 'currency', 'N/A'),
                                "exchange": getattr(instrument, 'exchange', 'N/A'),
                            }
                
                # Если не нашли точного совпадения с BBG FIGI, возвращаем None
                logger.warning(f"Не найден инструмент с точным совпадением тикера {ticker} и FIGI формата BBG...")
                return None
            else:
                logger.warning(f"Инструменты не найдены для запроса: {ticker}")
                return None
    except Exception as e:
        logger.error(f"Ошибка при поиске инструмента {ticker}: {e}")
        return None


def get_instrument_by_uid(token: str, uid: str) -> Optional[Dict[str, Any]]:
    """Получить информацию об инструменте по UID через find_instrument
    
    Args:
        token: Токен доступа к T-Invest API
        uid: UID инструмента
        
    Returns:
        Dict с информацией о тикере (ticker, figi, name, type) или None если не найден
    """
    try:
        with Client(token) as client:
            # Используем instruments.find_instrument для поиска инструмента по UID
            instruments_response = client.instruments.find_instrument(query=uid)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
                # Ищем инструмент с точным совпадением UID
                for instrument in instruments_response.instruments:
                    inst_uid = getattr(instrument, 'uid', None)
                    if inst_uid == uid:
                        figi = getattr(instrument, 'figi', None)
                        # Возвращаем только FIGI в формате BBG... (требуется для post_order)
                        if figi and figi.startswith('BBG'):
                            return {
                                "ticker": getattr(instrument, 'ticker', 'N/A'),
                                "figi": figi,
                                "name": getattr(instrument, 'name', 'N/A'),
                                "uid": inst_uid,
                                "type": getattr(instrument, 'instrument_kind', 'N/A'),
                                "currency": getattr(instrument, 'currency', 'N/A'),
                                "exchange": getattr(instrument, 'exchange', 'N/A'),
                            }
                
                logger.warning(f"Не найден инструмент с UID {uid} и FIGI формата BBG...")
                return None
            else:
                logger.warning(f"Инструменты не найдены для UID: {uid}")
                return None
    except Exception as e:
        logger.error(f"Ошибка при поиске инструмента по UID {uid}: {e}")
        return None


def get_instrument_by_figi(token: str, figi: str) -> Optional[Dict[str, Any]]:
    """Получить информацию об инструменте по FIGI через find_instrument
    
    Args:
        token: Токен доступа к T-Invest API
        figi: FIGI инструмента
        
    Returns:
        Dict с информацией о тикере (ticker, name, type, uid) или None если не найден
    """
    try:
        with Client(token) as client:
            # Используем instruments.find_instrument для поиска инструмента по FIGI
            instruments_response = client.instruments.find_instrument(query=figi)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
                # Ищем инструмент с точным совпадением FIGI
                for instrument in instruments_response.instruments:
                    inst_figi = getattr(instrument, 'figi', None)
                    if inst_figi == figi:
                        return {
                            "ticker": getattr(instrument, 'ticker', 'N/A'),
                            "figi": inst_figi,
                            "name": getattr(instrument, 'name', 'N/A'),
                            "uid": getattr(instrument, 'uid', None),
                            "type": getattr(instrument, 'instrument_kind', 'N/A'),
                            "currency": getattr(instrument, 'currency', 'N/A'),
                            "exchange": getattr(instrument, 'exchange', 'N/A'),
                        }
                
                logger.warning(f"Не найден инструмент с FIGI {figi}")
                return None
            else:
                logger.warning(f"Инструменты не найдены для FIGI: {figi}")
                return None
    except Exception as e:
        logger.error(f"Ошибка при поиске инструмента по FIGI {figi}: {e}")
        return None


def get_figi_by_ticker(token: str, ticker: str) -> Optional[str]:
    """Получить FIGI по тикуру через find_instrument
    
    Args:
        token: Токен доступа к T-Invest API
        ticker: Тикер инструмента (например, 'VTBR', 'SNGS') - точное совпадение
        
    Returns:
        FIGI в формате BBG... или None если не найден
    """
    info = get_instrument_by_ticker(token, ticker)
    if info:
        return info["figi"]
    return None
