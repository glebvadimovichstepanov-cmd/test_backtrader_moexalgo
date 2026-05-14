# Lag-Llama Trading System v2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MOEX](https://img.shields.io/badge/MOEX-API-green.svg)](https://www.moex.com/ru/algopack/about)

## 📖 Описание

**Lag-Llama Trading System** — это комплексная система алгоритмической торговли, объединяющая:

1. **Модель Lag-Llama** — современная модель машинного обучения для прогнозирования временных рядов
2. **MOEX API (moexalgo)** — получение биржевых данных с Московской Биржи
3. **T-Invest API** — исполнение заявок через брокера Т-Банк
4. **Графический интерфейс (GUI)** — удобное управление торговлей с визуализацией графиков

### Ключевые возможности

✅ **Генерация торговых сигналов** на основе прогнозов модели Lag-Llama  
✅ **Мультитаймфреймовый анализ** (10min, 1H, 1D) с весовыми коэффициентами  
✅ **Фильтрация сигналов** по confidence, R/R ratio, прогнозу profit  
✅ **Автоматическое выставление заявок** с Stop-Loss и Take-Profit  
✅ **Графический интерфейс** с отрисовкой графиков цены и уровней SL/TP  
✅ **Логирование операций** в реальном времени  
✅ **Поддержка Live-торговли** через T-Invest API

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Основные зависимости
pip install numpy pandas torch moexalgo requests websockets matplotlib

# Lag-Llama (опционально, для генерации сигналов)
pip install git+https://github.com/time-series-foundation-models/lag-llama.git

# T-Invest API (для Live-торговли)
pip install t_tech.invest

# Графический интерфейс (встроен в tkinter)
# Для Ubuntu/Debian: sudo apt-get install python3-tk
# Для Windows/macOS: входит в стандартную поставку Python
```

### 2. Настройка API токена

Получите токен в личном кабинете T-Invest: https://t-invest.ru/settings/api

```bash
# Linux/macOS
export INVEST_TOKEN='ваш_токен'

# Windows (PowerShell)
$env:INVEST_TOKEN='ваш_токен'

# Windows (CMD)
set INVEST_TOKEN=ваш_токен
```

### 3. Запуск

#### Вариант A: Графический интерфейс (рекомендуется)

```bash
python gui_app.py
```

#### Вариант B: Консольный режим

```bash
python main.py
```

---

## 📁 Структура проекта

```
/workspace/
├── gui_app.py                 # Графический интерфейс (Tkinter)
├── main.py                    # Основная логика интеграции
├── test_lag_llama.py          # Модель Lag-Llama и генерация сигналов
├── README.md                  # Документация
├── backtrader_moexalgo/       # Интеграция с Backtrader
│   ├── __init__.py
│   ├── moexalgo_store.py
│   └── moexalgo_feed.py
├── DataExamplesMoexAlgo_ru/   # Примеры работы с данными MOEX
├── StrategyExamplesMoexAlgo_ru/ # Примеры торговых стратегий
└── my_config/                 # Примеры конфигурационных файлов
```

---

## 🎯 Как это работает

### Архитектура системы

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Lag-Llama     │────▶│  Signal Generator │────▶│  Validation     │
│   Model         │     │  (test_lag_llama) │     │  Filters        │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   T-Invest API  │◀────│   Order Executor │◀────│   Signal OK?   │
│   (Live Trade)  │     │   (main.py)      │     │   (Y/N)        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         ▲
         │
┌─────────────────┐
│   GUI App       │
│   (gui_app.py)  │
└─────────────────┘
```

### Поток данных

1. **Запрос тикера**: Пользователь вводит тикер инструмента (SNGS, SBER, etc.)
2. **Загрузка данных**: Система загружает исторические данные с MOEX через moexalgo
3. **Прогнозирование**: Lag-Llama генерирует прогноз на нескольких таймфреймах
4. **Генерация сигнала**: Определяется направление (LONG/SHORT), SL, TP, confidence
5. **Валидация**: Проверка по критериям (confidence ≥ 60%, R/R ≥ 1.5, profit ≥ 0.9%)
6. **Исполнение**: Выставление заявки через T-Invest API с SL/TP

---

## ⚙️ Параметры и настройки

### Параметры сигнала (в gui_app.py)

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `MIN_CONFIDENCE` | 60% | Минимальная уверенность модели |
| `MIN_RR_RATIO` | 1.5 | Минимальное соотношение риск/прибыль |
| `MIN_PROFIT_PCT` | 0.9% | Минимальный прогнозируемый profit |
| `ENSEMBLE_RUNS` | 3 | Количество запусков модели для усреднения |

### Веса таймфреймов

```python
TF_WEIGHTS = {
    "10T": 0.50,  # 10 минут (основной для интрадей)
    "1H": 0.35,   # 1 час
    "1D": 0.15    # 1 день
}
```

### Торговая сессия MOEX

Система автоматически проверяет активную сессию:
- **Начало**: 10:30 MSK (не торгуем в первые 30 мин)
- **Конец**: 18:30 MSK (не торгуем в последние 30 мин)

---

## 🖥️ Графический интерфейс

### Основные компоненты

1. **Панель управления** (слева)
   - Поле ввода тикера
   - Кнопка "📊 Получить сигнал"
   - Кнопка "💰 Выставить заявку"
   - Настройки параметров (Min Confidence, R/R, Profit)

2. **График цены** (справа вверху)
   - Визуализация исторических данных
   - Отображение уровней входа, SL и TP
   - Интерактивные инструменты (масштабирование, сохранение)

3. **Панель сигнала** (справа по центру)
   - Тикер, направление, confidence
   - Цена входа, Stop Loss, Take Profit
   - R/R ratio, прогноз profit
   - Статус валидности сигнала

4. **Консоль логов** (справа внизу)
   - Логи операций в реальном времени
   - Цветовая индикация уровней (INFO/WARNING/ERROR)
   - Возможность сохранения в файл

### Скриншот интерфейса

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Lag-Llama Trading System v2.0                                          │
├──────────────────────┬──────────────────────────────────────────────────┤
│  Управление торговлей │  График цены и сигнал                           │
│                      │  ┌───────────────────────────────────────────┐  │
│  Тикер: [SNGS____]   │  │      ╭─╮                                  │  │
│                      │  │     ╱   ╲    ╭─╮                          │  │
│  [📊 Получить сигнал]│  │    ╱     ╲  ╱   ╲────── TP ────────────── │  │
│                      │  │   ╱       ╭╯     ╲                         │  │
│  [💰 Выставить заявку]│  │  ╱       ╱         ╲                       │  │
│                      │  │ ╱       ╱           ╲─── SL ────────────── │  │
│  Параметры сигнала:  │  └───────────────────────────────────────────┘  │
│  Min Confidence: 60  ├──────────────────────────────────────────────────┤
│  Min R/R:      1.5   │  Текущий сигнал                                  │
│  Min Profit:   0.9   │                                                  │
│                      │  Тикер:        SNGS                             │
│                      │  Направление:  LONG                             │
│                      │  Confidence:   75%                              │
│                      │  Цена входа:   25.4500                          │
│                      │  Stop Loss:    24.8000                          │
│                      │  Take Profit:  26.5000                          │
│                      │  R/R Ratio:    2.15                             │
│                      │  Прогноз:      +1.25%                           │
│                      │  Статус:       ✅ Готов к торговле              │
│                      ├──────────────────────────────────────────────────┤
│                      │  Лог операций                                    │
│                      │  [10:45:32] INFO: Генерация сигнала...          │
│                      │  [10:45:35] INFO: Сигнал получен                │
│                      │  [10:45:36] SUCCESS: Валидация OK               │
│                      │                                                  │
└──────────────────────┴──────────────────────────────────────────────────┘
│  Готов к работе | Сессия: активна                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Примеры использования

### Консольный режим

```bash
$ python main.py

=== Интеграция Lag-Llama + T-Bank Live Trading ===

Введите тикер инструмента (или 'exit' для выхода): SNGS

[INFO] Запрошен тикер: SNGS
[INFO] Генерация сигнала для SNGS...
[INFO] Сигнал: LONG | Confidence: 75/100 | SL: 24.80 | TP: 26.50 | R/R: 2.15
[INFO] Проверка сигнала: OK
[INFO] Поиск FIGI для SNGS...
[INFO] ✅ Найдено: SNGS (TQBR) -> BBG004MK7Z79
[INFO] Заявка успешно выставлена!
[INFO] SL и TP установлены.
```

### Программный вызов

```python
from main import generate_signal, validate_signal

# Генерация сигнала
signal = generate_signal("SNGS", logger)

if signal:
    is_valid, reason = validate_signal(signal)
    
    if is_valid:
        print(f"✅ Сигнал валиден: {signal['signal']}")
        print(f"   Confidence: {signal['confidence']}%")
        print(f"   SL: {signal['sl']}, TP: {signal['tp']}")
    else:
        print(f"❌ Сигнал отклонён: {reason}")
```

---

## 🔒 Безопасность

### Рекомендации

⚠️ **Важно**:
- Не храните токен в коде или репозитории
- Используйте переменные окружения для токена
- Проверяйте параметры перед выставлением заявки
- Тестируйте стратегию на демо-счете
- Не используйте публичные компьютеры для торговли

### Ограничения ответственности

Программа предоставляется **"КАК ЕСТЬ"** (AS IS). Автор не несет ответственности за:
- Финансовые потери при использовании
- Убытки от ошибок в коде или данных
- Последствия неправильной настройки

Торговля на бирже связана с риском потери капитала.

---

## 🛠️ Расширение функционала

### Добавление нового индикатора

```python
# В test_lag_llama.py добавьте:
def calculate_custom_indicator(data: pd.DataFrame) -> pd.Series:
    """Расчет пользовательского индикатора"""
    # Ваша логика
    return indicator_series
```

### Интеграция с другим брокером

```python
# Создайте файл broker_xxx.py по аналогии с main.py
from xxx_api import Client

def place_order_xxx(...):
    # Логика выставления заявок для вашего брокера
    pass
```

---

## 📚 Дополнительные ресурсы

### Документация

- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [MOEX API Docs](https://github.com/moexalgo/moexalgo)
- [T-Invest API](https://tinkoff.github.io/investAPI/)
- [Lag-Llama Paper](https://arxiv.org/abs/2310.08239)

### Видеоуроки

- [YouTube: Работа с библиотекой](https://youtu.be/SmcQF2jPxsQ)
- [RuTube: Начало работы](https://rutube.ru/video/private/ba9e19f36c98d45ac9caf5b399dda6ca/)

### Примеры стратегий

Смотрите папки:
- `DataExamplesMoexAlgo_ru/` — работа с данными
- `StrategyExamplesMoexAlgo_ru/` — торговые стратегии

---

## 🤝 Вклад в проект

Приветствуются:
- Pull Requests с улучшениями
- Bug reports
- Предложения по новым функциям
- Документация и примеры

```bash
git clone https://github.com/WISEPLAT/backtrader_moexalgo
cd backtrader_moexalgo
git checkout -b feature/your-feature-name
```

---

## 📄 Лицензия

MIT License — см. файл [LICENSE](LICENSE)

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WISEPLAT/backtrader_moexalgo&type=Timeline)](https://star-history.com/#WISEPLAT/backtrader_moexalgo&Timeline)

---

## 🙏 Благодарности

- **Backtrader** — отличная библиотека для бэктестинга
- **Команда MOEX** — за API moexalgo
- **Игорь Чечета** — за библиотеки FinamPy, AlorPy, QuikPy
- **Сообщество** — за вклад и поддержку

---

## 📞 Контакты

- GitHub: [WISEPLAT](https://github.com/WISEPLAT)
- Вопросы: создавайте Issue в репозитории

---

**Happy Trading! 📈💰**

## Установка
1) Самый простой способ:
```shell
pip install backtrader_moexalgo
```
или
```shell
git clone https://github.com/WISEPLAT/backtrader_moexalgo
```
или
```shell
pip install git+https://github.com/WISEPLAT/backtrader_moexalgo.git
```

2) Пожалуйста, используйте backtrader из моего репозитория (так как вы можете размещать в нем свои коммиты). Установите его:
```shell
pip install git+https://github.com/WISEPLAT/backtrader.git
```
-- Могу ли я использовать ваш интерфейс для moexalgo с оригинальным backtrader?

-- Да, вы можете использовать оригинальный backtrader, так как автор оригинального backtrader одобрил все мои изменения.

Вот ссылка: [mementum/backtrader#472](https://github.com/mementum/backtrader/pull/472)

3) У нас есть некоторые зависимости, вам нужно их установить:
```shell
pip install numpy pandas backtrader moexalgo requests websockets matplotlib
```

### Начало работы
Чтобы было легче разобраться как всё работает, сделано множество примеров в папках **DataExamplesMoexAlgo_ru** и **StrategyExamplesMoexAlgo_ru**.

```shell
Внимание! Для получения Super Candles - необходима авторизация на сайте Московской биржи.

1. Активируйте эту строку:
store = MoexAlgoStore(login=ConfigMOEX.Login, password=ConfigMOEX.Password)  # Хранилище AlgoPack + авторизация на Московской Бирже

2. Заполните ваши учетные данные в файле Config.py
class Config:
    Login = '<Адрес электронной почты>'  # Адрес электронной почты, указанный при регистрации на сайте moex.com
    Password = '<Пароль>'  # Пароль от учетной записи на сайте moex.com
```

В папке **DataExamplesMoexAlgo_ru** находится код примеров по работе с биржевыми данными через API интерфейс [MOEX](https://www.moex.com/ru/algopack/about ).

* **01 - Symbol.py** - торговая стратегия для получения исторических и "живых" данных одного тикера по одному таймфрейму
  * Реализована возможность получения Super Candles, свечей с расширенным набором параметров
* **02 - Symbol data to DF.py** - экспорт в csv файл исторических данных одного тикера по одному таймфрейму
* **03 - Symbols.py** - торговая стратегия для нескольких тикеров по одному таймфрейму
* **04 - Rollover.py** - запуск торговой стратегии на склейке данных из файла с историческими данными и последней загруженной истории с брокера
* **05 - Timeframes.py** - торговая стратегия для одного тикера по разным таймфреймам
* **Strategy.py** - Пример торговой стратегии, которая только выводит данные по тикеру/тикерам OHLCV

В папке **StrategyExamplesMoexAlgo_ru** находится код примеров стратегий.  

* **01 - Live Trade - broker Alor.py** - Пример торговой стратегии в live режиме для тикера SBER - брокер Алор. 
  * Пример выставления заявок на биржу через брокера Алор и их снятие.
    * Пожалуйста, имейте в виду! Это live режим - если на рынке произойдет значительное изменение цены в сторону понижения более чем на 0.5% - ордер может быть выполнен.... 
    * **Не забудьте после теста снять с биржи выставленные заявки!**
* **01 - Live Trade - broker Finam.py** - Пример торговой стратегии в live режиме для тикера SBER - брокер Финам. 
  * Пример выставления заявок на биржу через брокера Финам и их снятие.
    * Пожалуйста, имейте в виду! Это live режим - если на рынке произойдет значительное изменение цены в сторону понижения более чем на 0.5% - ордер может быть выполнен.... 
    * **Не забудьте после теста снять с биржи выставленные заявки!**
* **01 - Live Trade - brokers with Quik.py** - Пример торговой стратегии в live режиме для тикера SBER - любые брокеры с терминалом Quik. 
  * Пример выставления заявок на биржу через брокеров с терминалом Quik и их снятие.
    * Пожалуйста, имейте в виду! Это live режим - если на рынке произойдет значительное изменение цены в сторону понижения более чем на 0.5% - ордер может быть выполнен.... 
    * **Не забудьте после теста снять с биржи выставленные заявки!**

* **02 - Offline Backtest.py** - Пример торговой стратегии для теста на истории - не live режим - для двух тикеров SBER и LKOH.
  * В стратегии показано как применять индикаторы (SMA, RSI) к нескольким тикерам одновременно.
    * Не live режим - для тестирования стратегий без отправки заявок на биржу!
* **03 - Offline Backtest MultiPortfolio.py** - Пример торговой стратегии для теста на истории - не live режим - для множества тикеров, которые можно передавать в стратегию списком (SBER, LKOH, AFLT, GMKN). 
  * В стратегии показано как применять индикаторы (SMA, RSI) к нескольким тикерам одновременно.
    * Не live режим - для тестирования стратегий без отправки заявок на биржу!
* **04 - Offline Backtest Indicators.py** - Пример торговой стратегии для теста на истории с использованием индикаторов SMA и RSI - не live режим - для двух тикеров SBER и LKOH. 
  * В стратегии показано как применять индикаторы (SMA, RSI) к нескольким тикерам одновременно.
    * генерит 177% дохода на момент записи видео )) 
    * Не live режим - для тестирования стратегий без отправки заявок на биржу!

## Спасибо
- backtrader: очень простая и классная библиотека!
- Команде разработчиков MOEX [moexalgo](https://github.com/moexalgo/moexalgo): Для создания оболочки MOEX API, сокращающей большую часть работы.
- Игорю Чечету: за классные бесплатные библиотеки для live торговли реализующие подключения к брокерам 

## Важно
Исправление ошибок, доработка и развитие библиотеки осуществляется автором и сообществом!

**Пушьте ваши коммиты!** 

# Условия использования
Библиотека backtrader_moexalgo позволяющая делать интеграцию Backtrader и MOEX API - это **Программа** созданная исключительно для удобства работы.
При использовании **Программы** Пользователь обязан соблюдать положения действующего законодательства Российской Федерации или своей страны.
Использование **Программы** предлагается по принципу «Как есть» («AS IS»). Никаких гарантий, как устных, так и письменных не прилагается и не предусматривается.
Автор и сообщество не дает гарантии, что все ошибки **Программы** были устранены, соответственно автор и сообщество не несет никакой ответственности за
последствия использования **Программы**, включая, но, не ограничиваясь любым ущербом оборудованию, компьютерам, мобильным устройствам, 
программному обеспечению Пользователя вызванным или связанным с использованием **Программы**, а также за любые финансовые потери,
понесенные Пользователем в результате использования **Программы**.
Никто не ответственен за потерю данных, убытки, ущерб, включаю случайный или косвенный, упущенную выгоду, потерю доходов или любые другие потери,
связанные с использованием **Программы**.

**Программа** распространяется на условиях лицензии [MIT](https://choosealicense.com/licenses/mit).

## История звезд
Пожалуйста, поставьте Звезду 🌟 этому коду

[![Star History Chart](https://api.star-history.com/svg?repos=WISEPLAT/backtrader_moexalgo&type=Timeline)](https://star-history.com/#WISEPLAT/backtrader_moexalgo&Timeline)

## Star History
Please, put a Star 🌟 for this code

==========================================================================


# backtrader_moexalgo
MOEX API integration with [Backtrader](https://github.com/WISEPLAT/backtrader).
This code was written for [GO ALGO](https://goalgo.ru ) Hackathon, organized by the [MOEX](https://www.moex.com/ru/algopack/about ) exchange.

With this integration you can do:
 - Backtesting your strategy on historical data from the exchange [MOEX](https://www.moex.com/ru/algopack/about ) + [Backtrader](https://github.com/WISEPLAT/backtrader )  // Backtesting 
 - Launch trading systems for automatic trading on the exchange [MOEX](https://www.moex.com/ru/algopack/about) + [Backtrader](https://github.com/WISEPLAT/backtrader ) // Live trading
   - For Live trading you need install free additional libraries, like these:
     - broker Finam: [MOEX](https://www.moex.com/ru/algopack/about) + [Backtrader](https://github.com/WISEPLAT/backtrader ) +  [FinamPy](https://github.com/cia76/FinamPy ) // Live trading
     - broker Alor: [MOEX](https://www.moex.com/ru/algopack/about) + [Backtrader](https://github.com/WISEPLAT/backtrader ) +  [AlorPy](https://github.com/cia76/AlorPy ) // Live trading
     - ANY broker with the use of Quik terminal: [MOEX](https://www.moex.com/ru/algopack/about) + [Backtrader](https://github.com/WISEPLAT/backtrader ) +  [QuikPy](https://github.com/cia76/QuikPy ) // Live trading
 - Download live, historical data and Super Candles for tickers from the exchange [MOEX](https://www.moex.com/ru/algopack/about)
- Create and test your trading strategies using the library's features [Backtrader](https://github.com/WISEPLAT/backtrader )
  - There is a lot of useful documentation on how to make strategies [(see here)](https://www.backtrader.com/docu/quickstart/quickstart/ ).

For API connection we are using library [moexalgo](https://github.com/moexalgo/moexalgo).

**A tutorial video with this library can be viewed [on YouTube](https://youtu.be/SmcQF2jPxsQ ) and [on RuTube](https://rutube.ru/video/private/ba9e19f36c98d45ac9caf5b399dda6ca/?p=2T_4kwuwMz9aQAY3kFxYfQ )**

## Installation
1) The simplest way:
```shell
pip install backtrader_moexalgo
```
or
```shell
git clone https://github.com/WISEPLAT/backtrader_moexalgo
```
or
```shell
pip install git+https://github.com/WISEPLAT/backtrader_moexalgo.git
```

2) Please use backtrader from my repository (as your can push your commits in it). Install it:
```shell
pip install git+https://github.com/WISEPLAT/backtrader.git
```
-- Can I use your interface for moexalgo with original backtrader?

-- Yes, you can use original backtrader, as the author of original backtrader had approved all my changes. 

Here is the link: [mementum/backtrader#472](https://github.com/mementum/backtrader/pull/472)

3) We have some dependencies, you need to install them: 
```shell
pip install numpy pandas backtrader moexalgo requests websockets matplotlib
```

### Getting started
To make it easier to figure out how everything works, many examples have been made in the folders **DataExamplesMoexAlgo** and **StrategyExamplesMoexAlgo**.

```shell
Attention! To receive Super Candles, you need to log in to the Moscow Exchange website.

1. Activate this line:
store = MoexAlgoStore(login=ConfigMOEX.Login, password=ConfigMOEX.Password)  # Storage AlgoPack + authorization on the Moscow Stock Exchange

2. Fill in your credentials in the file Config.py
class Config:
    Login = '<Email address>'  # The email address provided during registration on the site moex.com
    Password = '<Password>'  # The password for the account on the site moex.com
```

The **DataExamplesMoexAlgo** folder contains the code of examples for working with exchange data via the [MOEX](https://www.moex.com/ru/algopack/about ) API.

* **01 - Symbol.py** - trading strategy for obtaining historical and "live" data of one ticker for one timeframe
  * The possibility of obtaining Super Candles, candles with an extended set of parameters is implemented
* **02 - Symbol data to DF.py** - export to csv file of historical data of one ticker for one timeframe
* **03 - Symbols.py** - trading strategy for multiple tickers on the same timeframe
* **04 - Rollover.py** - launch of a trading strategy based on gluing data from a file with historical data and the last downloaded history from the broker
* **05 - Timeframes.py** - trading strategy is running on different timeframes.
* **Strategy.py** - An example of a trading strategy that only outputs data of the OHLCV for ticker/tickers

The **StrategyExamplesMoexAlgo** folder contains the code of sample strategies.

* **01 - Live Trade - broker Alor.py** - An example of a live trading strategy for SBER ticker - broker Alor.
  * Example of placing and cancel orders on the exchange with the use of broker Alor.
    * Please be aware! This is Live order - if market has a big change down in value of price more than 0.5% - the order will be completed.... 
    * **Do not forget to cancel the submitted orders from the exchange after the test!**
* **01 - Live Trade - broker Finam.py** - An example of a live trading strategy for SBER ticker - broker Finam.
  * Example of placing and cancel orders on the exchange with the use of broker Finam.
    * Please be aware! This is Live order - if market has a big change down in value of price more than 0.5% - the order will be completed.... 
    * **Do not forget to cancel the submitted orders from the exchange after the test!**
* **01 - Live Trade - broker brokers with Quik.py** - An example of a live trading strategy for SBER ticker - any brokers with terminal Quik.
  * Example of placing and cancel orders on the exchange with the use of broker with terminal Quik.
    * Please be aware! This is Live order - if market has a big change down in value of price more than 0.5% - the order will be completed.... 
    * **Do not forget to cancel the submitted orders from the exchange after the test!**

* **02 - Offline Backtest.py** - An example of a trading strategy on a historical data - not live mode - for two SBER and LKOH tickers.
  * The strategy shows how to apply indicators (SMA, RSI) to several tickers at the same time.
    * Not a live mode - for testing strategies without sending orders to the exchange!
* **03 - Offline Backtest MultiPortfolio.py** - An example of a trading strategy on a historical data - not live mode - for a set of tickers that can be transferred to the strategy in a list (SBER, LKOH, AFLT, GMKN).
  * The strategy shows how to apply indicators (SMA, RSI) to several tickers at the same time.
    * Not a live mode - for testing strategies without sending orders to the exchange!
* **04 - Offline Backtest Indicators.py** - An example of a trading strategy for a history test using SMA and RSI indicators - not live mode - for two SBER and LKOH tickers.
  * The strategy shows how to apply indicators (SMA, RSI) to several tickers at the same time.
    * generates 177% of revenue at the time of video recording))
    * Non-live mode - for testing strategies without sending orders to the exchange!

## Thanks
- backtrader: Very simple and cool library!
- Team of MOEX [moexalgo](https://github.com/moexalgo/moexalgo): For creating MOEX API wrapper, shortening a lot of work.
- Igor Chechet: for free cool libraries for live trading by connection to brokers 

## License
[MIT](https://choosealicense.com/licenses/mit)

## Important
Error correction, revision and development of the library is carried out by the author and the community!

**Push your commits!**

## Terms of Use
The backtrader_moexalgo library, which allows you to integrate Backtrader and MOEX API, is the **Program** created solely for the convenience of work.
When using the **Program**, the User is obliged to comply with the provisions of the current legislation of his country.
Using the **Program** are offered on an "AS IS" basis. No guarantees, either oral or written, are attached and are not provided.
The author and the community does not guarantee that all errors of the **Program** have been eliminated, respectively, the author and the community do not bear any responsibility for
the consequences of using the **Program**, including, but not limited to, any damage to equipment, computers, mobile devices,
User software caused by or related to the use of the **Program**, as well as for any financial losses
incurred by the User as a result of using the **Program**.
No one is responsible for data loss, losses, damages, including accidental or indirect, lost profits, loss of revenue or any other losses
related to the use of the **Program**.

The **Program** is distributed under the terms of the [MIT](https://choosealicense.com/licenses/mit ) license.
