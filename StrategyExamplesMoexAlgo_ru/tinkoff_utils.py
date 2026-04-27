"""Utility functions for T-Invest API - getting instrument info by ticker/figi/uid."""
from typing import Optional, Dict, Any

from t_tech.invest import Client


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
            instruments_response = client.instruments.find_instrument(query=ticker)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
                for instrument in instruments_response.instruments:
                    inst_ticker = getattr(instrument, 'ticker', None)
                    if inst_ticker == ticker:
                        figi = getattr(instrument, 'figi', None)
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
                
                return None
            else:
                return None
    except Exception as e:
        print(f"Ошибка при поиске инструмента {ticker}: {e}")
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
            instruments_response = client.instruments.find_instrument(query=uid)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
                for instrument in instruments_response.instruments:
                    inst_uid = getattr(instrument, 'uid', None)
                    if inst_uid == uid:
                        figi = getattr(instrument, 'figi', None)
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
                
                return None
            else:
                return None
    except Exception as e:
        print(f"Ошибка при поиске инструмента по UID {uid}: {e}")
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
            instruments_response = client.instruments.find_instrument(query=figi)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
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
                
                return None
            else:
                return None
    except Exception as e:
        print(f"Ошибка при поиске инструмента по FIGI {figi}: {e}")
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
