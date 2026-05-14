#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Интеграция Lag-Llama сигнала с Live-торговлей T-Bank

Логика:
1. Запрос тикера у пользователя
2. Генерация сигнала через test_lag_llama.py
3. Фильтрация сигнала (confidence, R/R, сессия)
4. Выставка заявки через T-Invest API (логика из 01-LiveTrade-brokerTbank.py)
"""

import os
import sys
import re
import uuid
import logging
from datetime import datetime
from typing import Tuple, Optional

# Импорт функций из test_lag_llama.py
import test_lag_llama

# Импорт утилит из t_tech.invest
from t_tech.invest import Client, OrderDirection, OrderType

# Настройки
MIN_CONFIDENCE = 60
MIN_RR_RATIO = 1.5


def setup_logger() -> logging.Logger:
    """Настройка логгера в консоль + файл"""
    logger = logging.getLogger("main_integration")
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие хендлеры
    if logger.handlers:
        logger.handlers.clear()
    
    # Форматтер
    fmt = logging.Formatter(
        "[%(levelname)s] %(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Файловый хендлер
    fh = logging.FileHandler("main_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Консольный хендлер
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger


def get_user_ticker() -> str:
    """
    Интерактивный запрос тикера с валидацией.
    
    Returns:
        Тикер инструмента или пустая строка при выходе
    """
    print("\n=== Интеграция Lag-Llama + T-Bank Live Trading ===\n")
    
    while True:
        try:
            ticker = input("Введите тикер инструмента (или 'exit' для выхода): ").strip().upper()
            
            # Проверка на выход
            if ticker.lower() in ('exit', 'q', 'quit'):
                return ""
            
            # Проверка на пустую строку
            if not ticker:
                print("⚠️  Тикер не может быть пустым. Попробуйте снова.\n")
                continue
            
            # Валидация: только латиница и цифры
            if not re.match(r'^[A-Z0-9]+$', ticker):
                print("⚠️  Тикер должен содержать только латинские буквы и цифры. Попробуйте снова.\n")
                continue
            
            return ticker
            
        except KeyboardInterrupt:
            print("\n\nВвод отменён пользователем")
            return ""
        except EOFError:
            return ""


def generate_signal(ticker: str, logger: logging.Logger) -> Optional[dict]:
    """
    Вызвать main() из test_lag_llama.py и вернуть сигнал.
    
    Args:
        ticker: Тикер инструмента
        logger: Логгер для вывода
        
    Returns:
        dict с сигналом или None при ошибке
    """
    try:
        # Вызываем main() с параметрами:
        # - ticker: указываем нужный тикер
        # - return_signal=True: получаем dict вместо только логирования
        # - logger: передаём наш логгер для единого вывода
        signal = test_lag_llama.main(
            ticker=ticker,
            return_signal=True,
            logger=logger
        )
        
        if signal is None:
            logger.error("Функция main() вернула None")
            return None
        
        # Проверяем структуру сигнала
        required_keys = ["ticker", "signal", "confidence", "sl", "tp", "rr", "session_ok", "timestamp"]
        for key in required_keys:
            if key not in signal:
                logger.error(f"В сигнале отсутствует ключ '{key}'")
                return None
        
        return signal
        
    except Exception as e:
        logger.error(f"Ошибка при генерации сигнала: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def validate_signal(signal: dict) -> Tuple[bool, str]:
    """
    Проверка сигнала на соответствие критериям входа.
    
    Args:
        signal: dict с данными сигнала
        
    Returns:
        (is_valid, reason) — кортеж с результатом и причиной
    """
    # Проверка 1: наличие сигнала
    if signal["signal"] == "НЕТ СИГНАЛА" or signal["signal"].startswith("НЕТ СИГНАЛА"):
        return False, f"Нет торгового сигнала: {signal['signal']}"
    
    # Проверка 2: confidence >= порога
    if signal["confidence"] < MIN_CONFIDENCE:
        return False, f"Уверенность {signal['confidence']}% ниже порога {MIN_CONFIDENCE}%"
    
    # Проверка 3: R/R ratio >= минимума
    if signal["rr"] < MIN_RR_RATIO:
        return False, f"Risk/Reward {signal['rr']} ниже минимума {MIN_RR_RATIO}"
    
    # Проверка 4: активная сессия MOEX
    if not signal["session_ok"]:
        return False, "Торговая сессия MOEX не активна"
    
    return True, "OK"


def get_account_id(token: str, logger: logging.Logger) -> Optional[str]:
    """
    Получить ID активного счета.
    
    Args:
        token: Токен доступа к T-Invest API
        logger: Логгер
        
    Returns:
        account_id или None
    """
    try:
        with Client(token) as client:
            accounts_response = client.users.get_accounts()
            
            if not hasattr(accounts_response, 'accounts') or not accounts_response.accounts:
                logger.error("Список счетов пуст")
                return None
            
            from t_tech.invest import AccountStatus
            
            for acc in accounts_response.accounts:
                status_value = acc.status.value if hasattr(acc.status, 'value') else str(acc.status)
                logger.info(f"Проверка счета {acc.id}, статус: {status_value}")
                
                if acc.status == AccountStatus.ACCOUNT_STATUS_OPEN:
                    logger.info(f"Найден активный счет: {acc.id}")
                    return acc.id
            
            logger.error("Не найдено счетов со статусом ACCOUNT_STATUS_OPEN")
            return None
            
    except Exception as e:
        logger.error(f"Ошибка при получении списка счетов: {e}")
        return None


def get_figi_for_ticker(token: str, ticker: str, class_code: str = None, logger: logging.Logger = None) -> Optional[str]:
    """
    Получить FIGI для тикера через T-Invest API.
    
    Args:
        token: Токен доступа к T-Invest API
        ticker: Тикер инструмента (например, 'SNGS')
        class_code: Код режима торгов (например, 'TQBR'). Если None, используется первый найденный.
        logger: Логгер
        
    Returns:
        FIGI в формате BBG... или None если не найден
    """
    try:
        with Client(token) as client:
            instruments_response = client.instruments.find_instrument(query=ticker)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
                # Сначала ищем точное совпадение тикера с FIGI формата BBG
                for instrument in instruments_response.instruments:
                    inst_ticker = getattr(instrument, 'ticker', None)
                    figi = getattr(instrument, 'figi', None)
                    
                    if inst_ticker == ticker and figi and figi.startswith('BBG'):
                        if class_code is not None:
                            inst_class_code = getattr(instrument, 'class_code', None)
                            if inst_class_code == class_code:
                                return figi
                        else:
                            return figi
                
                # Если не нашли с class_code, пробуем без него
                if class_code is not None:
                    if logger:
                        logger.info(f"Не найден инструмент {ticker} с class_code={class_code}, пробуем без class_code")
                    for instrument in instruments_response.instruments:
                        inst_ticker = getattr(instrument, 'ticker', None)
                        figi = getattr(instrument, 'figi', None)
                        if inst_ticker == ticker and figi and figi.startswith('BBG'):
                            return figi
                
                if logger:
                    logger.error(f"Не найден инструмент с точным совпадением тикера {ticker} и FIGI формата BBG...")
                return None
            else:
                if logger:
                    logger.error(f"Инструменты не найдены для запроса: {ticker}")
                return None
                
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при поиске инструмента {ticker}: {e}")
        return None


def get_figi_with_moex_store(ticker: str, token: str, logger: logging.Logger) -> Optional[str]:
    """
    Получить FIGI через MoexAlgoStore + T-Invest API (как в 01-LiveTrade-brokerTbank.py).
    
    Args:
        ticker: Тикер инструмента
        token: Токен T-Invest API
        logger: Логгер
        
    Returns:
        FIGI или None
    """
    try:
        from backtrader_moexalgo.moexalgo_store import MoexAlgoStore
        
        store = MoexAlgoStore()
        info_df = store.get_symbol_info(ticker)
        
        # Ищем основную доску (is_primary == 1)
        primary_board = info_df[info_df['is_primary'] == 1]
        
        if len(primary_board) > 0:
            class_code = primary_board.iloc[0]['board']  # Например, 'TQBR'
            logger.info(f"✅ Тикер {ticker}: class_code={class_code} (основной режим торгов)")
            
            # Получаем FIGI через T-Invest API с учетом class_code
            figi = get_figi_for_ticker(token, ticker, class_code, logger)
            if figi:
                logger.info(f"✅ Найдено: {ticker} ({class_code}) -> {figi}")
                return figi
            else:
                logger.warning(f"❌ Не удалось получить FIGI для {ticker} (class_code={class_code})")
        else:
            logger.warning(f"⚠️ Тикер {ticker}: не найден основной режим торгов, пробуем без class_code")
        
        # Пробуем без class_code
        figi = get_figi_for_ticker(token, ticker, None, logger)
        if figi:
            logger.info(f"✅ Найдено: {ticker} -> {figi}")
            return figi
        
        logger.error(f"❌ Не удалось получить FIGI для {ticker}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Ошибка при получении информации о тикере {ticker}: {e}")
        # Пробуем напрямую через T-Invest API
        figi = get_figi_for_ticker(token, ticker, None, logger)
        if figi:
            logger.info(f"✅ Найдено через T-Invest API: {ticker} -> {figi}")
            return figi
        return None


def place_order(signal: dict, figi: str, token: str, account_id: str, logger: logging.Logger) -> bool:
    """
    Выставить заявку по сигналу.
    
    Args:
        signal: dict с данными сигнала
        figi: FIGI инструмента
        token: Токен T-Invest API
        account_id: ID счета
        logger: Логгер
        
    Returns:
        True если заявка успешно выставлена, иначе False
    """
    # Определяем направление
    if signal["signal"] == "LONG":
        direction = OrderDirection.ORDER_DIRECTION_BUY
        direction_str = "BUY"
    elif signal["signal"] == "SHORT":
        direction = OrderDirection.ORDER_DIRECTION_SELL
        direction_str = "SELL"
    else:
        logger.error(f"Неизвестный сигнал: {signal['signal']}")
        return False
    
    ticker = signal["ticker"]
    logger.info(f"Выставка заявки {direction_str} 1 лот {ticker} по рынку...")
    
    try:
        with Client(token) as client:
            response = client.orders.post_order(
                instrument_id=figi,
                quantity=1,
                direction=direction,
                account_id=account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
                order_id=str(uuid.uuid4()),
            )
            
            order_id = response.order_id
            logger.info(f"Заявка исполнена: order_id={order_id}")
            logger.info(f"Статус: {response}")
            return True
            
    except Exception as e:
        error_msg = str(e)
        
        if "90001" in error_msg or "Need confirmation" in error_msg:
            logger.error("Ошибка 90001: Требуется подтверждение сделки!")
            logger.error("   Откройте приложение Т-Инвестиций и подтвердите сессию/сделку.")
            logger.error("   После подтверждения перезапустите скрипт.")
            
        elif "50002" in error_msg or "Instrument not found" in error_msg:
            logger.error("Ошибка 50002: Инструмент не найден!")
            logger.error(f"   Тикер: {ticker}, FIGI: {figi}")
            logger.error("   Попробуйте использовать другой инструмент.")
            
        elif "30052" in error_msg or "Instrument forbidden for trading by API" in error_msg:
            logger.error("Ошибка 30052: Инструмент запрещен для торговли через API!")
            logger.error(f"   Тикер: {ticker}, FIGI: {figi}")
            logger.error("Возможные причины:")
            logger.error("   - Ограничения брокера или ЦБ РФ для данного инструмента")
            logger.error("   - Требуется расширенный статус квалифицированного инвестора")
            logger.error("   - Проверьте настройки профиля в приложении Т-Инвестиций")
            logger.error("   - Попробуйте другие инструменты (VTBR, GAZP, SBER)")
            
        else:
            logger.error(f"Ошибка при выставлении заявки: {e}")
        
        return False


def main():
    """Основной пайплайн"""
    logger = setup_logger()
    
    # Шаг 1: Запрос тикера
    ticker = get_user_ticker()
    if not ticker:
        logger.info("Ввод отменён")
        return
    
    logger.info(f"Запрошен тикер: {ticker}")
    
    # Шаг 2: Генерация сигнала
    logger.info(f"Генерация сигнала для {ticker}...")
    signal = generate_signal(ticker, logger)
    
    if not signal:
        logger.error("Не удалось получить сигнал")
        return
    
    logger.info(
        f"Сигнал: {signal['signal']} | Confidence: {signal['confidence']}/100 | "
        f"SL: {signal['sl']} | TP: {signal['tp']} | R/R: {signal['rr']} | "
        f"Сессия: {'активна' if signal['session_ok'] else 'закрыта'}"
    )
    
    # Шаг 3: Валидация сигнала
    is_valid, reason = validate_signal(signal)
    if not is_valid:
        logger.warning(f"Сигнал отклонён: {reason}")
        return
    
    logger.info(f"Проверка сигнала: OK (confidence={signal['confidence']} >= {MIN_CONFIDENCE}, "
                f"R/R={signal['rr']} >= {MIN_RR_RATIO}, сессия активна)")
    
    # Шаг 4: Получение токена и account_id
    token = os.getenv('INVEST_TOKEN')
    if not token:
        logger.error("INVEST_TOKEN не установлен")
        logger.error("Установите переменную окружения перед запуском:")
        logger.error("  Windows (PowerShell): $env:INVEST_TOKEN='ваш_токен'")
        logger.error("  Linux/Mac: export INVEST_TOKEN='ваш_токен'")
        return
    
    logger.info("Токен T-Invest найден")
    
    # Пробуем получить account_id из переменной окружения или через API
    account_id = os.getenv('INVEST_ACCOUNT_ID')
    if not account_id:
        logger.info("Получение account_id через API...")
        account_id = get_account_id(token, logger)
        if not account_id:
            logger.error("Не удалось получить account_id")
            return
    else:
        logger.info(f"account_id получен из переменной окружения: {account_id}")
    
    # Шаг 5: Поиск FIGI
    logger.info(f"Поиск FIGI для {ticker}...")
    figi = get_figi_with_moex_store(ticker, token, logger)
    
    if not figi:
        logger.error(f"Не найден FIGI для {ticker}")
        return
    
    # Шаг 6: Выставление заявки
    success = place_order(signal, figi, token, account_id, logger)
    
    if success:
        logger.info("=== Завершение: Заявка успешно выставлена ===")
    else:
        logger.warning("=== Завершение: Заявка не выставлена ===")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
