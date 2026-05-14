# Промпт для создания main.py — Интеграция Lag-Llama сигнала с Live-торговлей T-Bank

## Задача
Создать скрипт `main.py`, который объединяет два существующих скрипта:
1. `test_lag_llama.py` — генерирует торговый сигнал на основе ML-прогноза
2. `StrategyExamplesMoexAlgo_ru/01-LiveTrade-brokerTbank.py` — выставляет заявки через T-Invest API

## Требования к функционалу

### 1. Интерактивный ввод тикера
- Скрипт запрашивает у пользователя тикер инструмента (например, `SNGS`, `VTBR`, `GAZP`)
- Валидация ввода: проверка на пустую строку, допустимые символы (латиница, цифры)
- Возможность выхода по `Ctrl+C` или вводу `exit`/`q`

### 2. Генерация сигнала через test_lag_llama.py
После ввода тикера:
- Динамически изменить глобальную переменную `TICKER` в `test_lag_llama.py` перед запуском
- Импортировать и вызвать основную функцию `main()` из `test_lag_llama.py` программно (не через subprocess)
- **Важно:** Модифицировать `test_lag_llama.py` так, чтобы:
  - Функция `main()` принимала аргумент `ticker` (по умолчанию `None`, использует глобальный `TICKER`)
  - Функция `main()` возвращала структурированный результат сигнала (dict) вместо только логирования
  - Сохранить обратную совместимость при запуске как `python test_lag_llama.py`

**Структура возвращаемого сигнала:**
```python
{
    "ticker": str,           # тикер инструмента
    "signal": str,           # "LONG", "SHORT", или "НЕТ СИГНАЛА"
    "confidence": int,       # 0-100
    "sl": float,             # стоп-лосс
    "tp": float,             # тейк-профит
    "rr": float,             # risk/reward ratio
    "session_ok": bool,      # активна ли сессия MOEX
    "timestamp": str         # время генерации сигнала
}
```

### 3. Фильтрация сигнала перед торговлей
Перед выставлением заявки проверить:
- `signal != "НЕТ СИГНАЛА"` — иначе завершить с сообщением
- `confidence >= 60` (порог из `test_lag_llama.py`)
- `rr >= 1.5` (минимальный R/R ratio)
- `session_ok == True` — торгуем только в активную сессию MOEX

Если проверки не пройдены → вывести причину и завершить.

### 4. Выставление заявки через логику 01-LiveTrade-brokerTbank.py
Использовать ключевые функции из `01-LiveTrade-brokerTbank.py`:
- `get_tinkoff_token()` — получение токена из `INVEST_TOKEN`
- `get_account_id(token)` — получение ID счёта
- `get_figi_for_ticker(token, ticker, class_code=None)` — поиск FIGI инструмента
- `execute_with_client(func)` — хелпер для операций с клиентом

**Логика выставления заявки:**
```python
if signal["signal"] == "LONG":
    direction = OrderDirection.ORDER_DIRECTION_BUY
elif signal["signal"] == "SHORT":
    direction = OrderDirection.ORDER_DIRECTION_SELL
else:
    return  # нет сигнала

# Получить FIGI через MoexAlgoStore + T-Invest API (как в 01-LiveTrade-brokerTbank.py)
# Выставить заявку по рынку на 1 лот:
response = client.orders.post_order(
    instrument_id=figi,
    quantity=1,
    direction=direction,
    account_id=account_id,
    order_type=OrderType.ORDER_TYPE_MARKET,
    order_id=str(uuid.uuid4())
)
```

**Важно:** 
- Не использовать полный класс `RSIStrategy` из `01-LiveTrade-brokerTbank.py` (там демо-логика с покупкой/продажей через 3 бара)
- Использовать только утилиты для работы с T-Invest API и моновыставление заявки по сигналу Lag-Llama
- Обработать ошибки: 90001 (подтверждение), 50002 (инструмент), 30052 (запрещён API)

### 5. Логирование и вывод
- Вывод в консоль каждого этапа:
  1. Запрос тикера
  2. Запуск генерации сигнала
  3. Результат сигнала (direction, confidence, SL, TP, R/R)
  4. Проверки фильтра
  5. Поиск FIGI
  6. Результат выставки заявки (order_id, статус)
- Логирование в файл `main_log.txt` (дубль консоли)
- При ошибке — понятное сообщение с рекомендацией (как в `01-LiveTrade-brokerTbank.py`)

### 6. Структура main.py
```python
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
import uuid
import logging
from datetime import datetime

# Импорт функций из test_lag_llama.py
# (потребуется модификация test_lag_llama.py для возврата сигнала)

# Импорт утилит из 01-LiveTrade-brokerTbank.py
from t_tech.invest import Client, OrderDirection, OrderType
# ... другие импорты

# Настройки
MIN_CONFIDENCE = 60
MIN_RR_RATIO = 1.5

def setup_logger():
    """Настройка логгера в консоль + файл"""
    ...

def get_user_ticker() -> str:
    """Интерактивный запрос тикера с валидацией"""
    ...

def generate_signal(ticker: str) -> dict:
    """
    Вызвать main() из test_lag_llama.py и вернуть сигнал.
    Требуется модификация test_lag_llama.py:
    - main(ticker=None) принимает тикер
    - main() возвращает dict сигнала
    """
    ...

def validate_signal(signal: dict) -> tuple[bool, str]:
    """Проверка сигнала на соответствие критериям входа"""
    ...

def get_instrument_figi(ticker: str, token: str) -> str | None:
    """
    Получить FIGI инструмента через MoexAlgoStore + T-Invest API.
    Логика из 01-LiveTrade-brokerTbank.py: start() метода RSIStrategy
    """
    ...

def place_order(signal: dict, figi: str, token: str, account_id: str):
    """Выставить заявку по сигналу"""
    ...

def main():
    """Основной пайплайн"""
    logger = setup_logger()
    
    # Шаг 1: Запрос тикера
    ticker = get_user_ticker()
    if not ticker:
        logger.info("Ввод отменён")
        return
    
    # Шаг 2: Генерация сигнала
    logger.info(f"Генерация сигнала для {ticker}...")
    signal = generate_signal(ticker)
    if not signal:
        logger.error("Не удалось получить сигнал")
        return
    
    logger.info(f"Сигнал: {signal['signal']} | Confidence: {signal['confidence']}/100 | "
                f"SL: {signal['sl']} | TP: {signal['tp']} | R/R: {signal['rr']}")
    
    # Шаг 3: Валидация сигнала
    is_valid, reason = validate_signal(signal)
    if not is_valid:
        logger.warning(f"Сигнал отклонён: {reason}")
        return
    
    # Шаг 4: Получение токена и account_id
    token = os.getenv('INVEST_TOKEN')
    if not token:
        logger.error("INVEST_TOKEN не установлен")
        return
    
    account_id = os.getenv('INVEST_ACCOUNT_ID')
    if not account_id:
        account_id = get_account_id(token)
        if not account_id:
            logger.error("Не удалось получить account_id")
            return
    
    # Шаг 5: Поиск FIGI
    figi = get_instrument_figi(ticker, token)
    if not figi:
        logger.error(f"Не найден FIGI для {ticker}")
        return
    
    # Шаг 6: Выставление заявки
    place_order(signal, figi, token, account_id)
    
    logger.info("=== Завершение ===")

if __name__ == "__main__":
    main()
```

### 7. Необходимые изменения в test_lag_llama.py
Для интеграции потребуется модифицировать `test_lag_llama.py`:
1. Добавить параметр `ticker` в функцию `main(ticker: str = None)`
2. Если `ticker` передан — использовать его вместо глобального `TICKER`
3. В конце `main()` вернуть dict сигнала вместо `None`
4. Обернуть финальный `print` и логирование, но не удалять (для обратной совместимости)

Пример изменения:
```python
def main(ticker: str = None) -> Optional[dict]:
    global TICKER
    if ticker:
        TICKER = ticker
    
    # ... существующий код ...
    
    signal_data = {
        "ticker": TICKER,
        "signal": signal_text,
        "direction": consensus_dir,
        "confidence": confidence,
        "sl": final_sl,
        "tp": final_tp,
        "rr": final_rr,
        "session_ok": session_ok,
        "timestamp": timestamp
    }
    
    # Сохранение и логирование остаются
    save_signal(signal_data, timestamp)
    
    return signal_data  # Новый возврат
```

### 8. Зависимости
Убедиться, что установлены:
```bash
pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
pip install moexalgo backtrader gluonts torch huggingface_hub
```

### 9. Переменные окружения
Перед запуском установить:
```bash
export INVEST_TOKEN='ваш_токен_t_invest'
export INVEST_ACCOUNT_ID='опционально_ID_счёта'
```

### 10. Пример сессии пользователя
```
$ python main.py

=== Интеграция Lag-Llama + T-Bank Live Trading ===

Введите тикер инструмента (или 'exit' для выхода): SNGS

[INFO] Генерация сигнала для SNGS...
[INFO] Загрузка данных с MOEX (1D, 1H, 10T)...
[INFO] Прогноз Lag-Llama (ensemble x3)...
[INFO] Сигнал: LONG | Confidence: 75/100 | SL: 28.50 | TP: 31.20 | R/R: 1.8

[INFO] Проверка сигнала: OK (confidence=75 >= 60, R/R=1.8 >= 1.5, сессия активна)
[INFO] Поиск FIGI для SNGS...
[INFO] Найдено: SNGS (TQBR) -> BBG004MKCLT8

[INFO] Выставка заявки BUY 1 лот SNGS по рынку...
[INFO] Заявка исполнена: order_id=550e8400-e29b-41d4-a716-446655440000

=== Завершение ===
```

## Критерии приёмки
- [ ] Скрипт запрашивает тикер и корректно обрабатывает ввод
- [ ] Вызывает `test_lag_llama.py` программно с указанным тикером
- [ ] Получает структурированный сигнал (dict)
- [ ] Фильтрует сигнал по confidence, R/R, сессии
- [ ] Находит FIGI через MoexAlgoStore + T-Invest API
- [ ] Выставляет заявку через T-Invest API (BUY для LONG, SELL для SHORT)
- [ ] Обрабатывает ошибки API с понятными сообщениями
- [ ] Логирует все этапы в консоль и файл
- [ ] Обратная совместимость: `test_lag_llama.py` работает как standalone
