#Документация T-Invest API  https://opensource.tbank.ru/invest/invest-python
# pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

import datetime as dt
from datetime import timezone
import os
import uuid
import backtrader as bt
from backtrader_moexalgo.moexalgo_store import MoexAlgoStore  # Хранилище AlgoPack

# пример live торговли для Tbank (Тинькофф Инвестиции)
from t_tech.invest import Client  # Коннект к Tbank API - для выставления заявок на покупку/продажу
from t_tech.invest import OrderDirection, OrderType, TimeInForceType, InstrumentType

# Токен берем из переменной окружения INVEST_TOKEN
INVEST_TOKEN = os.getenv('INVEST_TOKEN')
if not INVEST_TOKEN:
    raise ValueError("Переменная окружения INVEST_TOKEN не установлена. Установите её перед запуском скрипта.\n"
                     "Пример установки:\n"
                     "  Windows (PowerShell): $env:INVEST_TOKEN='ваш_токен'\n"
                     "  Linux/Mac: export INVEST_TOKEN='ваш_токен'")

# ID счета берем из переменной окружения INVEST_ACCOUNT_ID (опционально)
INVEST_ACCOUNT_ID = os.getenv('INVEST_ACCOUNT_ID', None)

from Config import Config as ConfigMOEX  # для авторизации на Московской Бирже

# Импортируем утилиты для работы с инструментами
from tinkoff_utils import InstrumentCache


# Глобальный токен для работы с Tinkoff API
_tinkoff_token = None
# Глобальный кэш инструментов
_instrument_cache = None

def get_tinkoff_token():
    """Получить токен Tinkoff API"""
    global _tinkoff_token
    if _tinkoff_token is None:
        _tinkoff_token = INVEST_TOKEN
    return _tinkoff_token


def get_instrument_cache():
    """Получить кэш инструментов (создает при первом вызове)"""
    global _instrument_cache
    if _instrument_cache is None:
        _instrument_cache = InstrumentCache(INVEST_TOKEN)
    return _instrument_cache

def get_account_id(token):
    """Получить ID активного счета"""
    try:
        with Client(token) as client:
            accounts_response = client.users.get_accounts()
            print(f"Получены счета: {accounts_response}")
            if not hasattr(accounts_response, 'accounts') or not accounts_response.accounts:
                print("Внимание: список счетов пуст или имеет неверный формат")
                return None
            for acc in accounts_response.accounts:
                # Проверяем статус счета - ищем открытый счет
                # AccountStatus.ACCOUNT_STATUS_OPEN = 2
                from t_tech.invest import AccountStatus
                status_value = acc.status.value if hasattr(acc.status, 'value') else str(acc.status)
                print(f"Проверка счета {acc.id}, статус: {status_value} ({acc.status})")
                if acc.status == AccountStatus.ACCOUNT_STATUS_OPEN:
                    print(f"Найден активный счет: {acc.id}")
                    return acc.id
            print("Не найдено счетов со статусом ACCOUNT_STATUS_OPEN")
            return None
    except Exception as e:
        print(f"Ошибка при получении списка счетов: {e}")
        return None


def get_figi_for_ticker(token, ticker, class_code=None):
    """Получить FIGI для тикера через T-Invest API
    
    Args:
        token: Токен доступа к T-Invest API
        ticker: Тикер инструмента (например, 'SNGS')
        class_code: Код режима торгов/доски (например, 'TQBR'). Если None, используется первый найденный.
    
    Returns:
        FIGI в формате BBG... или None если не найден
    """
    try:
        with Client(token) as client:
            # Используем instruments.find_instrument для поиска инструмента по тикуру
            instruments_response = client.instruments.find_instrument(query=ticker)
            
            if hasattr(instruments_response, 'instruments') and instruments_response.instruments:
                # Ищем инструмент с FIGI формата BBG... (требуется для post_order)
                for instrument in instruments_response.instruments:
                    figi = getattr(instrument, 'figi', None)
                    
                    # Возвращаем только FIGI в формате BBG... (требуется для post_order)
                    if figi and figi.startswith('BBG'):
                        return figi
                
                # Если не нашли BBG FIGI, возвращаем None
                return None
            else:
                return None
    except Exception as e:
        print(f"Ошибка при поиске инструмента {ticker}: {e}")
        return None


def print_positions_info(token, account_id):
    """Вывод информации о позициях на счете: тикеры, FIGI, лоты, стоимость
    
    Args:
        token: Токен доступа к T-Invest API
        account_id: ID счета
    """
    try:
        # Получаем кэш инструментов для быстрого поиска по UID/FIGI
        cache = get_instrument_cache()
        
        with Client(token) as client:
            # Получаем портфель
            portfolio_response = client.operations.get_portfolio(account_id=account_id)
            
            print("\n" + "="*70)
            print("ПОЗИЦИИ НА СЧЕТЕ {}".format(account_id))
            print("="*70)
            
            if not hasattr(portfolio_response, 'positions') or not portfolio_response.positions:
                print("Позиции не найдены")
                print("="*70 + "\n")
                return
            
            total_value = 0.0
            
            for pos in portfolio_response.positions:
                # Получаем информацию об инструменте
                instrument_uid = getattr(pos, 'instrument_uid', None)
                figi_from_pos = getattr(pos, 'figi', None)
                
                quantity = getattr(pos, 'quantity', None)
                if quantity and hasattr(quantity, 'units'):
                    nano = getattr(quantity, 'nano', 0)
                    lots = float(quantity.units) + float(nano) / 1e9
                else:
                    lots = 0.0
                
                # Получаем текущую цену
                current_price = getattr(pos, 'current_price', None)
                price_value = 0.0
                currency = 'RUB'
                if current_price:
                    units = getattr(current_price, 'units', 0)
                    nano = getattr(current_price, 'nano', 0)
                    price_value = float(units) + float(nano) / 1e9
                    curr_obj = getattr(current_price, 'currency', None)
                    if curr_obj:
                        currency = str(curr_obj)
                
                position_value = lots * price_value
                total_value += position_value
                
                # Пытаемся получить тикер и FIGI через кэш инструментов
                ticker = 'N/A'
                figi = 'N/A'
                
                # Сначала пробуем найти по UID (это наиболее надежный идентификатор в новом API)
                if instrument_uid:
                    inst_info = cache.get_by_uid(instrument_uid)
                    if inst_info:
                        ticker = inst_info.get('ticker', 'N/A')
                        figi = inst_info.get('figi', 'N/A')
                
                # Если не нашли по UID, пробуем по FIGI из позиции
                if ticker == 'N/A' and figi_from_pos:
                    inst_info = cache.get_by_figi(figi_from_pos)
                    if inst_info:
                        ticker = inst_info.get('ticker', 'N/A')
                        figi = figi_from_pos
                
                # Если все еще не нашли, пробуем старый метод через find_instrument
                if ticker == 'N/A':
                    try:
                        search_id = instrument_uid if instrument_uid else figi_from_pos
                        if search_id:
                            inst_response = client.instruments.find_instrument(query=search_id)
                            if hasattr(inst_response, 'instruments') and inst_response.instruments:
                                inst = inst_response.instruments[0]
                                ticker = getattr(inst, 'ticker', 'N/A')
                                figi = getattr(inst, 'figi', search_id)
                    except Exception as e:
                        logger.debug(f"Не удалось найти инструмент через find_instrument: {e}")
                
                print(f"Тикер: {ticker:10} | FIGI: {figi:20} | Лотов: {lots:12.2f} | Цена: {price_value:12.4f} {currency:4} | Стоимость: {position_value:12.2f} {currency}")
            
            print("-"*70)
            print(f"ОБЩАЯ СТОИМОСТЬ ПОЗИЦИЙ: {total_value:.2f} RUB")
            print("="*70 + "\n")
            
    except Exception as e:
        print(f"Ошибка при получении информации о позициях: {e}")
        import traceback
        traceback.print_exc()


def execute_with_client(func):
    """Хелпер для выполнения операций с клиентом в контекстном менеджере"""
    token = get_tinkoff_token()
    with Client(token) as client:
        return func(client)


# Торговая система
class RSIStrategy(bt.Strategy):
    """
    Демонстрация live стратегии - однократно покупаем по рынку 1 лот и однократно продаем его по рынку через 3 бара
    """
    params = (  # Параметры торговой системы
        ('timeframe', ''),
        ('live_prefix', ''),  # префикс для выставления заявок в live
        ('info_tickers', []),  # информация по тикерам
        ('tb_client', ''),  # Коннект к Tbank API - для выставления заявок на покупку/продажу
        ('account_id', ''),  # id счета
    )

    def __init__(self):
        """Инициализация, добавление индикаторов для каждого тикера"""
        self.orders = {}  # Организовываем заявки в виде справочника, конкретно для этой стратегии один тикер - одна активная заявка
        for d in self.datas:  # Пробегаемся по всем тикерам
            self.orders[d._name] = None  # Заявки по тикеру пока нет

        # создаем индикаторы для каждого тикера
        self.sma1 = {}
        self.sma2 = {}
        self.rsi = {}
        for i in range(len(self.datas)):
            ticker = list(self.dnames.keys())[i]    # key name is ticker name
            self.sma1[ticker] = bt.indicators.SMA(self.datas[i], period=8)  # SMA indicator
            self.sma2[ticker] = bt.indicators.SMA(self.datas[i], period=16)  # SMA indicator
            self.rsi[ticker] = bt.indicators.RSI(self.datas[i], period=14)  # RSI indicator

        self.buy_once = {}
        self.sell_once = {}
        self.order_time = None
        self.account_id = self.p.account_id
        self.client = None

    def start(self):
        for d in self.datas:  # Running through all the tickers
            self.buy_once[d._name] = False
            self.sell_once[d._name] = False
        
        # Получаем account_id если он не задан
        if not self.account_id:
            self.account_id = get_account_id(INVEST_TOKEN)
            if not self.account_id:
                raise ValueError("Не удалось получить account_id. Проверьте токен и статус счетов.")
        
        # Выводим информацию о позициях на счете при инициализации
        print_positions_info(INVEST_TOKEN, self.account_id)
        
        # Получаем FIGI для каждого тикера через store.get_symbol_info()
        # Формат: self.figi = self.store.provider.get_symbol_info(self.class_code, self.symbol).figi
        self.ticker_to_figi = {}
        for d in self.datas:
            ticker = d._name
            # Для moexalgo store используем get_symbol_info который возвращает DataFrame с информацией
            # Находим основной режим торгов (is_primary=1) и берем board (class_code)
            try:
                store = MoexAlgoStore()
                info_df = store.get_symbol_info(ticker)
                
                # Ищем основную доску (is_primary == 1)
                primary_board = info_df[info_df['is_primary'] == 1]
                
                if len(primary_board) > 0:
                    class_code = primary_board.iloc[0]['board']  # Например, 'TQBR'
                    print(f"✅ Тикер {ticker}: class_code={class_code} (основной режим торгов)")
                    
                    # Теперь получаем FIGI через T-Invest API с учетом class_code
                    figi = get_figi_for_ticker(INVEST_TOKEN, ticker, class_code)
                    if figi:
                        self.ticker_to_figi[ticker] = figi
                        print(f"✅ Тикер {ticker} (class_code={class_code}) -> FIGI: {figi}")
                    else:
                        print(f"❌ Не удалось получить FIGI для {ticker} (class_code={class_code})")
                        self.ticker_to_figi[ticker] = None
                else:
                    # Если не нашли основной режим, пробуем без class_code
                    print(f"⚠️ Тикер {ticker}: не найден основной режим торгов, пробуем без class_code")
                    figi = get_figi_for_ticker(INVEST_TOKEN, ticker)
                    if figi:
                        self.ticker_to_figi[ticker] = figi
                        print(f"✅ Тикер {ticker} -> FIGI: {figi}")
                    else:
                        print(f"❌ Не удалось получить FIGI для {ticker}")
                        self.ticker_to_figi[ticker] = None
            except Exception as e:
                print(f"❌ Ошибка при получении информации о тикере {ticker}: {e}")
                self.ticker_to_figi[ticker] = None
        
        # Получаем начальный баланс с брокера
        self._update_broker_balance()

    def _update_broker_balance(self):
        """Получить актуальный баланс свободного капитала от брокера"""
        try:
            def get_portfolio(client):
                # В новом API t_tech.invest используем client.operations.get_portfolio
                return client.operations.get_portfolio(account_id=self.account_id)
            
            portfolio_response = execute_with_client(get_portfolio)
            
            # PortfolioResponse имеет total_amount_currencies - это сумма всех валютных позиций
            # Для получения свободных денег используем total_amount_currencies
            self.broker_balance = 0.0
            
            if hasattr(portfolio_response, 'total_amount_currencies'):
                total_currencies = portfolio_response.total_amount_currencies
                if hasattr(total_currencies, 'units'):
                    nano = getattr(total_currencies, 'nano', 0)
                    self.broker_balance = float(total_currencies.units) + float(nano) / 1e9
            
            # Если balance все еще 0, пробуем найти RUB позицию в positions
            if self.broker_balance == 0.0 and hasattr(portfolio_response, 'positions'):
                for pos in portfolio_response.positions:
                    # instrument_type это строка, сравниваем напрямую
                    inst_type = getattr(pos, 'instrument_type', '')
                    # Проверяем тип инструмента - валюта (INSTRUMENT_TYPE_CURRENCY = 3)
                    if inst_type == 'currency' or inst_type == InstrumentType.INSTRUMENT_TYPE_CURRENCY or \
                       (hasattr(inst_type, 'value') and inst_type.value == 3):
                        # Проверяем currency внутри average_position_price или current_price
                        avg_price = getattr(pos, 'average_position_price', None)
                        if avg_price and hasattr(avg_price, 'currency'):
                            curr = avg_price.currency
                            # Ищем RUB или RUR позиции
                            if curr in ['RUB', 'RUR']:
                                quantity = getattr(pos, 'quantity', None)
                                if quantity and hasattr(quantity, 'units'):
                                    nano = getattr(quantity, 'nano', 0)
                                    self.broker_balance = float(quantity.units) + float(nano) / 1e9
                                    break
            
            print(f"Баланс получен от брокера: {self.broker_balance:.2f}")
        except Exception as e:
            print(f"Ошибка при получении баланса от брокера: {e}")
            self.broker_balance = 10000.0  # Значение по умолчанию

    def next(self):
        """Приход нового бара тикера"""
        for data in self.datas:  # Пробегаемся по всем запрошенным барам всех тикеров
            ticker = data._name
            status = data._state  # 0 - Live data, 1 - History data, 2 - None
            _interval = self.p.timeframe
            _date = bt.num2date(data.datetime[0])

            try:
                if data.p.supercandles[ticker][data.p.metric_name]:
                    print("\tSuper Candle:", data.p.supercandles[ticker][data.p.metric_name][0])
                    _data = data.p.supercandles[ticker][data.p.metric_name][0]
                    _data['datetime'] = _date
                    self.supercandles[ticker][data.p.metric_name].append(_data)
            except:
                pass

            if status in [0, 1]:
                if status: _state = "False - History data"
                else: _state = "True - Live data"

                print('{} / {} [{}] - Open: {}, High: {}, Low: {}, Close: {}, Volume: {} - Live: {}'.format(
                    bt.num2date(data.datetime[0]),
                    data._name,
                    _interval,  # таймфрейм тикера
                    data.open[0],
                    data.high[0],
                    data.low[0],
                    data.close[0],
                    data.volume[0],
                    _state,
                ))
                print(f'\t - {ticker} RSI : {self.rsi[ticker][0]}')

                if status != 0: continue  # если не live - то не входим в позицию!

                # Обновляем баланс от брокера перед использованием
                self._update_broker_balance()
                print(f"\t - Free balance: {self.broker_balance}")

                if not self.buy_once[ticker]:  # Enter long
                    free_money = self.broker_balance
                    print(f" - free_money: {free_money}")
                    print(f" - account_id: {self.account_id}")

                    # Получаем FIGI из предварительно заполненного словаря
                    instrument_id = self.ticker_to_figi.get(ticker)
                    
                    if not instrument_id:
                        print(f"❌ Пропуск заявки для {ticker}: не найден FIGI")
                        continue
                    
                    print(f" - instrument_id: {instrument_id} (FIGI из store.get_symbol_info)")

                    # Выставляем заявку на покупку по рынку
                    # Документация T-Invest API: https://opensource.tbank.ru/invest/invest-python
                    # Для маркет-ордеров time_in_force не требуется
                    def post_buy_order(client):
                        print(f"\n[DEBUG] POST /orders/postOrder:")
                        print(f"  instrument_id: {instrument_id}")
                        print(f"  quantity: 1")
                        print(f"  direction: ORDER_DIRECTION_BUY")
                        print(f"  account_id: {self.account_id}")
                        print(f"  order_type: ORDER_TYPE_MARKET")
                        return client.orders.post_order(
                            instrument_id=instrument_id,
                            quantity=1,
                            direction=OrderDirection.ORDER_DIRECTION_BUY,
                            account_id=self.account_id,
                            order_type=OrderType.ORDER_TYPE_MARKET,
                            order_id=str(uuid.uuid4()),
                        )
                    
                    try:
                        response = execute_with_client(post_buy_order)
                    except Exception as e:
                        error_msg = str(e)
                        if "90001" in error_msg or "Need confirmation" in error_msg:
                            print(f"⚠️ Ошибка 90001: Требуется подтверждение сделки!")
                            print("   Откройте приложение Т-Инвестиций и подтвердите сессию/сделку.")
                            print("   После подтверждения перезапустите скрипт.")
                        elif "50002" in error_msg or "Instrument not found" in error_msg:
                            print(f"⚠️ Ошибка 50002: Инструмент не найден!")
                            print(f"   Тикер: {ticker}, instrument_id: {instrument_id}")
                            print("   Попробуйте использовать FIGI или UID инструмента вместо тикера.")
                        elif "30052" in error_msg or "Instrument forbidden for trading by API" in error_msg:
                            print(f"⚠️ Ошибка 30052: Инструмент запрещен для торговли через API!")
                            print(f"   Тикер: {ticker}, instrument_id: {instrument_id}")
                            print("   Возможные причины:")
                            print("   - Ограничения брокера или ЦБ РФ для данного инструмента")
                            print("   - Требуется расширенный статус квалифицированного инвестора")
                            print("   - Проверьте настройки профиля в приложении Т-Инвестиций")
                            print("   - Попробуйте другие инструменты (VTBR, GAZP, SBER)")
                        else:
                            print(f"Ошибка при выставлении заявки: {e}")
                        continue
                    self.order_time = dt.datetime.now()
                    print(f"Выставили заявку на покупку 1 лота {ticker}:", response)
                    print("\t - order_id:", response.order_id)
                    print("\t - время:", self.order_time)

                    # print(f"\t - Выставлена заявка {self.orders[data._name]} на покупку {data._name}")

                    self.buy_once[ticker] = len(self)  # для однократной покупки + записываем номер бара

                else:  # Если есть позиция, т.к. покупаем сразу по рынку
                    print(self.sell_once[ticker], self.buy_once[ticker], len(self), len(self) > self.buy_once[ticker] + 3)
                    if not self.sell_once[ticker]:  # если мы еще не продаём
                        if self.buy_once[ticker] and len(self) > self.buy_once[ticker] + 3:  # если у нас есть позиция на 3-м баре после покупки
                            print("sell")
                            print(f"\t - Продаём по рынку {data._name}...")

                            # Получаем FIGI из предварительно заполненного словаря
                            instrument_id = self.ticker_to_figi.get(ticker)
                            
                            if not instrument_id:
                                print(f"❌ Пропуск заявки на продажу для {ticker}: не найден FIGI")
                                continue
                            
                            print(f" - instrument_id: {instrument_id} (FIGI из store.get_symbol_info)")

                            # Выставляем заявку на продажу по рынку
                            # Документация T-Invest API: https://opensource.tbank.ru/invest/invest-python
                            def post_sell_order(client):
                                print(f"\n[DEBUG] POST /orders/postOrder:")
                                print(f"  instrument_id: {instrument_id}")
                                print(f"  quantity: 1")
                                print(f"  direction: ORDER_DIRECTION_SELL")
                                print(f"  account_id: {self.account_id}")
                                print(f"  order_type: ORDER_TYPE_MARKET")
                                return client.orders.post_order(
                                    instrument_id=instrument_id,
                                    quantity=1,
                                    direction=OrderDirection.ORDER_DIRECTION_SELL,
                                    account_id=self.account_id,
                                    order_type=OrderType.ORDER_TYPE_MARKET,
                                    order_id=str(uuid.uuid4()),
                                )
                            
                            try:
                                response = execute_with_client(post_sell_order)
                            except Exception as e:
                                error_msg = str(e)
                                if "90001" in error_msg or "Need confirmation" in error_msg:
                                    print(f"⚠️ Ошибка 90001: Требуется подтверждение сделки!")
                                    print("   Откройте приложение Т-Инвестиций и подтвердите сессию/сделку.")
                                    print("   После подтверждения перезапустите скрипт.")
                                elif "50002" in error_msg or "Instrument not found" in error_msg:
                                    print(f"⚠️ Ошибка 50002: Инструмент не найден!")
                                    print(f"   Тикер: {ticker}, instrument_id: {instrument_id}")
                                    print("   Попробуйте использовать FIGI или UID инструмента вместо тикера.")
                                elif "30052" in error_msg or "Instrument forbidden for trading by API" in error_msg:
                                    print(f"⚠️ Ошибка 30052: Инструмент запрещен для торговли через API!")
                                    print(f"   Тикер: {ticker}, instrument_id: {instrument_id}")
                                    print("   Возможные причины:")
                                    print("   - Ограничения брокера или ЦБ РФ для данного инструмента")
                                    print("   - Требуется расширенный статус квалифицированного инвестора")
                                    print("   - Проверьте настройки профиля в приложении Т-Инвестиций")
                                    print("   - Попробуйте другие инструменты (VTBR, GAZP, SBER)")
                                else:
                                    print(f"Ошибка при выставлении заявки: {e}")
                                continue
                            self.order_time = None

                            print(f"Выставили заявку на продажу 1 лота {ticker}:", response)
                            print("\t - order_id:", response.order_id)
                            print("\t - время:", self.order_time)

                            self.sell_once[ticker] = True  # для предотвращения повторной продажи

    def notify_order(self, order):
        """Изменение статуса заявки"""
        order_data_name = order.data._name  # Имя тикера из заявки
        print("*"*50)
        self.log(f'Заявка номер {order.ref} {order.info["order_number"]} {order.getstatusname()} {"Покупка" if order.isbuy() else "Продажа"} {order_data_name} {order.size} @ {order.price}')
        if order.status == bt.Order.Completed:  # Если заявка полностью исполнена
            if order.isbuy():  # Заявка на покупку
                self.log(f'Покупка {order_data_name} Цена: {order.executed.price:.2f}, Объём: {order.executed.value:.2f}, Комиссия: {order.executed.comm:.2f}')
            else:  # Заявка на продажу
                self.log(f'Продажа {order_data_name} Цена: {order.executed.price:.2f}, Объём: {order.executed.value:.2f}, Комиссия: {order.executed.comm:.2f}')
                self.orders[order_data_name] = None  # Сбрасываем заявку на вход в позицию
        print("*" * 50)

    def notify_trade(self, trade):
        """Изменение статуса позиции"""
        if trade.isclosed:  # Если позиция закрыта
            self.log(f'Прибыль по закрытой позиции {trade.getdataname()} Общая={trade.pnl:.2f}, Без комиссии={trade.pnlcomm:.2f}')

    def log(self, txt, dt=None):
        """Вывод строки с датой на консоль"""
        dt = bt.num2date(self.datas[0].datetime[0]) if not dt else dt  # Заданная дата или дата текущего бара
        print(f'{dt.strftime("%d.%m.%Y %H:%M")}, {txt}')  # Выводим дату и время с заданным текстом на консоль

    def format_price(self, ticker, price):
        """
        Функция округления до шага цены step, сохраняя signs знаков после запятой
        print(round_custom_f(0.022636, 0.000005, 6)) --> 0.022635
        """
        step = self.p.info_tickers[ticker]['securities']['MINSTEP']  # сохраняем минимальный Шаг цены
        signs = self.p.info_tickers[ticker]['securities']['DECIMALS']  # сохраняем Кол-во десятичных знаков

        val = round(price / step) * step
        return float(("{0:." + str(signs) + "f}").format(val))


def get_some_info_for_tickers(tickers, live_prefix):
    """Функция для получения информации по тикерам"""
    info = {}
    for ticker in tickers:
        i = store.get_symbol_info(ticker)
        info[f"{live_prefix}{ticker}"] = i
    return info


if __name__ == '__main__':

    # брокер Финам[FinamPy]: git clone https://github.com/cia76/FinamPy
    # брокер Алор[AlorPy]: git clone https://github.com/cia76/AlorPy
    # брокер Тинькофф[Tbank/t-tech-investments]: pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
    # ЛЮБОЙ брокер у которого есть терминал Quik[QuikPy]: git clone https://github.com/cia76/QuikPy

    # пример для Tbank (Тинькофф)
    live_prefix = ''  # префикс для выставления заявок в live
    tb_client = Client  # Коннект к Tbank API - для выставления заявок на покупку/продажу

    # Получаем account_id из переменной окружения или используем None (будет выбран первый доступный)
    account_id = INVEST_ACCOUNT_ID
    
    # Инициализируем глобальный токен и получаем account_id при старте
    _ = get_tinkoff_token()  # Инициализация токена
    if not account_id:
        account_id = get_account_id(INVEST_TOKEN)
        if not account_id:
            raise ValueError("Не удалось получить account_id. Проверьте токен и статус счетов.")

    symbol = 'SNGS'  # Тикер в формате <Код тикера>
    # symbol2 = 'LKOH'  # Тикер в формате <Код тикера>

    store = MoexAlgoStore()  # Хранилище AlgoPack
    # store = MoexAlgoStore(login=ConfigMOEX.Login, password=ConfigMOEX.Password)  # Хранилище AlgoPack + авторизация на Московской Бирже

    cerebro = bt.Cerebro(quicknotify=True)  # Инициируем "движок" BackTrader

    # live подключение к брокеру будем делать напрямую

    # ----------------------------------------------------
    # Внимание! - Теперь это Live режим работы стратегии #
    # ----------------------------------------------------

    info_tickers = get_some_info_for_tickers([symbol, ], live_prefix)  # берем информацию о тикере (минимальный шаг цены, кол-во знаков после запятой)

    # live 1-минутные бары / таймфрейм M1
    timeframe = "M1"
    fromdate = dt.datetime.now(timezone.utc)
    data = store.getdata(timeframe=bt.TimeFrame.Minutes, compression=1, dataname=symbol, fromdate=fromdate,
                         live_bars=True, name=f"{live_prefix}{symbol}")  # поставьте здесь True - если нужно получать live бары # name - нужен для выставления в live заявок
    # data2 = store.getdata(timeframe=bt.TimeFrame.Minutes, compression=1, dataname=symbol2, fromdate=fromdate, live_bars=True)  # поставьте здесь True - если нужно получать live бары

    cerebro.adddata(data)  # Добавляем данные
    # cerebro.adddata(data2)  # Добавляем данные

    cerebro.addstrategy(RSIStrategy, timeframe=timeframe, live_prefix=live_prefix, info_tickers=info_tickers,
                        tb_client=tb_client,
                        account_id=account_id)  # Добавляем торговую систему

    cerebro.run()  # Запуск торговой системы
    # cerebro.plot()  # Рисуем график - в live режиме не нужно
