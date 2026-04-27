# pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

import datetime as dt
from datetime import timezone
import os
import uuid
import backtrader as bt
from backtrader_moexalgo.moexalgo_store import MoexAlgoStore  # Хранилище AlgoPack

# пример live торговли для Tbank (Тинькофф Инвестиции)
from t_tech.invest import Client  # Коннект к Tbank API - для выставления заявок на покупку/продажу
from t_tech.invest import OrderDirection, OrderType, TimeInForceType

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


# Глобальный клиент для работы с Tinkoff API
_tinkoff_client = None

def get_tinkoff_client():
    """Получить единый инстанс клиента Tinkoff API"""
    global _tinkoff_client
    if _tinkoff_client is None:
        _tinkoff_client = Client(INVEST_TOKEN)
    return _tinkoff_client

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
                status_value = acc.status.value if hasattr(acc.status, 'value') else str(acc.status)
                print(f"Проверка счета {acc.id}, статус: {status_value}")
                if status_value == 'ACCOUNT_STATUS_OPEN':
                    print(f"Найден активный счет: {acc.id}")
                    return acc.id
            print("Не найдено счетов со статусом ACCOUNT_STATUS_OPEN")
            return None
    except Exception as e:
        print(f"Ошибка при получении списка счетов: {e}")
        return None


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
        
        # Инициализируем клиент при старте стратегии
        self.client = get_tinkoff_client()
        
        # Получаем account_id если он не задан
        if not self.account_id:
            self.account_id = get_account_id(INVEST_TOKEN)
            if not self.account_id:
                raise ValueError("Не удалось получить account_id. Проверьте токен и статус счетов.")

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

                print(f"\t - Free balance: {self.broker.getcash()}")

                if not self.buy_once[ticker]:  # Enter long
                    free_money = self.broker.getcash()
                    print(f" - free_money: {free_money}")
                    print(f" - account_id: {self.account_id}")

                    # Выставляем заявку на покупку по рынку
                    response = self.client.orders.post_order(
                        instrument_id=ticker,
                        quantity=1,
                        direction=OrderDirection.ORDER_DIRECTION_BUY,
                        account_id=self.account_id,
                        order_type=OrderType.ORDER_TYPE_MARKET,
                        order_id=str(uuid.uuid4()),
                        time_in_force=TimeInForceType.TIME_IN_FORCE_DAY,
                    )
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

                            # Выставляем заявку на продажу по рынку
                            response = self.client.orders.post_order(
                                instrument_id=ticker,
                                quantity=1,
                                direction=OrderDirection.ORDER_DIRECTION_SELL,
                                account_id=self.account_id,
                                order_type=OrderType.ORDER_TYPE_MARKET,
                                order_id=str(uuid.uuid4()),
                                time_in_force=TimeInForceType.TIME_IN_FORCE_DAY,
                            )
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
    
    # Инициализируем глобальный клиент и получаем account_id при старте
    _ = get_tinkoff_client()  # Инициализация клиента
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
