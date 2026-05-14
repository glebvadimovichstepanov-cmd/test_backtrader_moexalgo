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
from t_tech.invest import Client, OrderDirection, OrderType, StopOrderDirection

# Настройки
MIN_CONFIDENCE = 60
MIN_RR_RATIO = 1.5
MIN_PROFIT_PCT = 0.9  # Минимальный прогноз profit в % для входа


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
        required_keys = ["ticker", "signal", "confidence", "sl", "tp", "rr", "session_ok", "timestamp", "predicted_profit_pct"]
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
    
    # Проверка 5: прогноз profit >= минимального порога
    if signal.get("predicted_profit_pct", 0) < MIN_PROFIT_PCT:
        return False, f"Прогноз profit {signal.get('predicted_profit_pct', 0):.2f}% ниже минимума {MIN_PROFIT_PCT}%"
    
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
    Выставить заявку по сигналу и установить SL/TP.
    
    Args:
        signal: dict с данными сигнала
        figi: FIGI инструмента
        token: Токен T-Invest API
        account_id: ID счета
        logger: Логгер
        
    Returns:
        True если заявка успешно выставлена, иначе False
    """
    from t_tech.invest import StopOrderDirection, StopOrderType, StopOrderExpirationType, Quotation
    
    # Определяем направление
    if signal["signal"] == "LONG":
        direction = OrderDirection.ORDER_DIRECTION_BUY
        direction_str = "BUY"
        stop_direction = StopOrderDirection.STOP_ORDER_DIRECTION_SELL
    elif signal["signal"] == "SHORT":
        direction = OrderDirection.ORDER_DIRECTION_SELL
        direction_str = "SELL"
        stop_direction = StopOrderDirection.STOP_ORDER_DIRECTION_BUY
    else:
        logger.error(f"Неизвестный сигнал: {signal['signal']}")
        return False
    
    ticker = signal["ticker"]
    
    # Оптимизация: выставляем лимитную заявку по цене 0.99 * price (на 1% лучше рынка)
    # Получаем текущую цену для расчета лимита
    try:
        with Client(token) as client:
            # Получаем последнюю цену через get_candles или last_price
            # Для простоты используем market data
            from t_tech.invest import CandleInterval
            from datetime import datetime, timedelta
            
            # Запрашиваем последние свечи за 1 минуту для получения текущей цены
            candles_response = client.market_data.get_candles(
                figi=figi,
                from_=datetime.now() - timedelta(minutes=1),
                to=datetime.now(),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN
            )
            
            if hasattr(candles_response, 'candles') and candles_response.candles:
                last_candle = candles_response.candles[-1]
                current_price = last_candle.close.units + last_candle.close.nano / 1e9
            else:
                # Если нет свечей, будем использовать рыночный ордер
                current_price = None
                
        if current_price:
            # Лимитная цена: 0.99 * current_price для BUY, 1.01 * current_price для SELL
            if direction == OrderDirection.ORDER_DIRECTION_BUY:
                limit_price = current_price * 0.99
                logger.info(f"Выставка лимитной заявки {direction_str} 1 лот {ticker} по цене {limit_price:.4f} (0.99 * market)")
            else:
                limit_price = current_price * 1.01
                logger.info(f"Выставка лимитной заявки {direction_str} 1 лот {ticker} по цене {limit_price:.4f} (1.01 * market)")
        else:
            logger.info(f"Не удалось получить текущую цену, выставляем по рынку...")
            limit_price = None
            
    except Exception as e:
        logger.warning(f"Ошибка при получении цены: {e}, выставляем по рынку")
        limit_price = None
    
    try:
        with Client(token) as client:
            if limit_price:
                # Лимитная заявка
                def make_quotation(price: float) -> Quotation:
                    units = int(price)
                    nano = int((price - units) * 1e9)
                    return Quotation(units=units, nano=nano)
                
                response = client.orders.post_order(
                    instrument_id=figi,
                    quantity=1,
                    direction=direction,
                    account_id=account_id,
                    order_type=OrderType.ORDER_TYPE_LIMIT,
                    price=make_quotation(limit_price),
                    order_id=str(uuid.uuid4()),
                )
            else:
                # Рыночная заявка
                response = client.orders.post_order(
                    instrument_id=figi,
                    quantity=1,
                    direction=direction,
                    account_id=account_id,
                    order_type=OrderType.ORDER_TYPE_MARKET,
                    order_id=str(uuid.uuid4()),
                )
            
            order_id = response.order_id
            exec_status = response.execution_report_status
            
            # Статусы: 1=FILL, 2=PARTIALLY_FILL, 3=REJECTED, 4=CANCELLED, 5=NEW, 6=PENDING_NEW, etc.
            if exec_status in [1, 2]:
                logger.info(f"Заявка исполнена: order_id={order_id}")
                exec_price = response.executed_order_price.units + response.executed_order_price.nano / 1e9
                logger.info(f"Цена исполнения: {exec_price:.4f}")
                
                # Выставляем SL и TP
                place_stop_orders(client, figi, account_id, signal, exec_price, stop_direction, logger)
                return True
            elif limit_price and exec_status in [5, 6]:  # NEW or PENDING_NEW
                logger.info(f"Лимитная заявка размещена и ожидает исполнения: order_id={order_id}, статус={exec_status}")
                logger.info(f"Лимитная цена: {limit_price:.4f}")
                # Для лимитной заявки не выставляем SL/TP сразу, так как она еще не исполнена
                return True
            else:
                logger.warning(f"Заявка не исполнилась мгновенно. Статус: {exec_status}")
                return False
            
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


def place_stop_orders(client, figi: str, account_id: str, signal: dict, entry_price: float, 
                      stop_direction: StopOrderDirection, logger: logging.Logger):
    """
    Выставить Stop-Loss и Take-Profit заявки.
    
    Args:
        client: T-Invest клиент
        figi: FIGI инструмента
        account_id: ID счета
        signal: dict с данными сигнала
        entry_price: Цена входа в позицию
        stop_direction: Направление стоп-заявки
        logger: Логгер
    """
    from t_tech.invest import StopOrderType, StopOrderExpirationType, Quotation
    
    sl_price = signal["sl"]
    tp_price = signal["tp"]
    ticker = signal["ticker"]
    
    # Для LONG: stop_direction = SELL (продажа для закрытия)
    # Для SHORT: stop_direction = BUY (покупка для закрытия)
    is_long = stop_direction == StopOrderDirection.STOP_ORDER_DIRECTION_SELL
    
    if is_long:
        # LONG позиция: SL ниже, TP выше
        # Добавляем небольшой запас к цене активации для надежности
        sl_activation_price = sl_price * 1.01  # +1% для активации SL
        tp_activation_price = tp_price * 0.99   # -1% для активации TP
        order_type_name = "Stop-Loss (продажа)"
    else:
        # SHORT позиция: SL выше, TP ниже
        sl_activation_price = sl_price * 0.99   # -1% для активации SL
        tp_activation_price = tp_price * 1.01   # +1% для активации TP
        order_type_name = "Stop-Loss (покупка)"
    
    logger.info(f"Выставление защитных ордеров для {ticker}:")
    logger.info(f"  Цена входа: {entry_price:.4f}")
    logger.info(f"  SL: {sl_price:.4f} -> активация по {sl_activation_price:.4f}")
    logger.info(f"  TP: {tp_price:.4f} -> активация по {tp_activation_price:.4f}")
    
    def make_quotation(price: float) -> Quotation:
        """Создать Quotation из float цены"""
        units = int(price)
        nano = int((price - units) * 1e9)
        return Quotation(units=units, nano=nano)
    
    try:
        # Выставление Stop-Loss
        logger.info(f"  → Выставление {order_type_name}...")
        sl_response = client.stop_orders.post_stop_order(
            figi=figi,
            quantity=1,
            price=0,  # Исполнение по рынку при активации
            stop_price=make_quotation(sl_activation_price),
            direction=stop_direction,
            account_id=account_id,
            expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
            stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
            expire_date=None,
            order_id=str(uuid.uuid4())
        )
        logger.info(f"  ✓ Stop-Loss установлен: id={sl_response.stop_order_id}")
        
        # Выставление Take-Profit
        tp_type_name = "Take-Profit (продажа)" if is_long else "Take-Profit (покупка)"
        logger.info(f"  → Выставление {tp_type_name}...")
        tp_response = client.stop_orders.post_stop_order(
            figi=figi,
            quantity=1,
            price=0,  # Исполнение по рынку при активации
            stop_price=make_quotation(tp_activation_price),
            direction=stop_direction,
            account_id=account_id,
            expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
            stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
            expire_date=None,
            order_id=str(uuid.uuid4())
        )
        logger.info(f"  ✓ Take-Profit установлен: id={tp_response.stop_order_id}")
        
    except Exception as e:
        logger.error(f"  ✗ Ошибка при выставлении SL/TP: {e}")
        logger.error("  Проверьте права доступа токена на работу со стоп-заявками.")


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
        f"Прогноз profit: {signal.get('predicted_profit_pct', 0):+.2f}% | "
        f"Сессия: {'активна' if signal['session_ok'] else 'закрыта'}"
    )
    
    # Шаг 3: Валидация сигнала
    is_valid, reason = validate_signal(signal)
    if not is_valid:
        logger.warning(f"Сигнал отклонён: {reason}")
        return
    
    logger.info(f"Проверка сигнала: OK (confidence={signal['confidence']} >= {MIN_CONFIDENCE}, "
                f"R/R={signal['rr']} >= {MIN_RR_RATIO}, profit>0.9%, сессия активна)")
    
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
