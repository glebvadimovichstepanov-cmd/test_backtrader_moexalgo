import backtrader as bt
import math


class StrategyMCWithTPSL(bt.Strategy):
    """
    Стратегия на пересечении скользящих средних (SMA 8 и 32) с установленными:
    - Take Profit: +1.8%
    - Stop Loss: -0.9%
    """
    params = (
        ('name', None),
        ('symbols', None),
        ('timeframe', ''),
        ('take_profit_pct', 1.8),   # Тейк-профит в процентах
        ('stop_loss_pct', 0.9),     # Стоп-лосс в процентах
    )

    def log(self, txt, dt=None):
        """Вывод строки с датой на консоль"""
        dt = bt.num2date(self.datas[0].datetime[0]) if not dt else dt
        print(f'{dt.strftime("%d.%m.%Y %H:%M")}, {txt}')

    def __init__(self):
        """Инициализация, добавление индикаторов для каждого тикера"""
        self.isLive = False

        self.orders = {}
        for d in self.datas:
            self.orders[d._name] = None

        # Индикаторы для каждого тикера
        self.sma1 = {}
        self.sma2 = {}
        self.crossover = {}
        self.crossdown = {}
        for i in range(len(self.datas)):
            ticker = list(self.dnames.keys())[i]
            self.sma1[ticker] = bt.indicators.SMA(self.datas[i], period=8)
            self.sma2[ticker] = bt.indicators.SMA(self.datas[i], period=32)

            # Сигнал на покупку: быстрая SMA пересекает медленную снизу вверх
            self.crossover[ticker] = bt.ind.CrossOver(self.sma1[ticker], self.sma2[ticker])

            # Сигнал на продажу: быстрая SMA пересекает медленную сверху вниз
            self.crossdown[ticker] = bt.ind.CrossOver(self.sma2[ticker], self.sma1[ticker])

    def next(self):
        """Приход нового бара тикера"""
        for data in self.datas:
            if not self.p.symbols or data._name in self.p.symbols:
                ticker = data.p.dataname
                self.log(f'{ticker} - {bt.TimeFrame.Names[data.p.timeframe]} {data.p.compression} - Open={data.open[0]:.2f}, High={data.high[0]:.2f}, Low={data.low[0]:.2f}, Close={data.close[0]:.2f}, Volume={data.volume[0]:.0f}',
                         bt.num2date(data.datetime[0]))

                _date = bt.num2date(data.datetime[0])

                try:
                    if data.p.supercandles[ticker][data.p.metric_name]:
                        print("\tSuper Candle:", data.p.supercandles[ticker][data.p.metric_name][0])
                        _data = data.p.supercandles[ticker][data.p.metric_name][0]
                        _data['datetime'] = _date
                        self.supercandles[ticker][data.p.metric_name].append(_data)
                except:
                    pass

                # Дополнительная нагрузка удалена для ускорения бэктеста
                # for i in range(1, 1000000):
                #     z = math.sqrt(64 * 64 * 64 * 64 * 64)

                # Сигналы - используем правильное обращение к линиям индикатора
                signal1 = self.crossover[ticker].crossover[0]  # Покупка
                signal2 = self.crossdown[ticker].crossover[0]  # Продажа

                if not self.getposition(data):  # Если позиции нет
                    if signal1 == 1:
                        free_money = self.broker.getcash()
                        price = data.close[0]
                        size = (free_money / price) * 0.25  # 25% от средств

                        # Расчет уровней TP и SL
                        tp_price = price * (1 + self.p.take_profit_pct / 100.0)
                        sl_price = price * (1 - self.p.stop_loss_pct / 100.0)

                        print("-" * 50)
                        print(f"\t - BUY {ticker} size = {size} at price = {price}")
                        print(f"\t - TP = {tp_price:.2f} (+{self.p.take_profit_pct}%)")
                        print(f"\t - SL = {sl_price:.2f} (-{self.p.stop_loss_pct}%)")

                        # Выставляем заявку на покупку с привязанными ордерами TP и SL
                        buy_order = self.buy(data=data, exectype=bt.Order.Limit, price=price, size=size)
                        
                        # Создаем заявки на выход (будут активированы после исполнения покупки)
                        # Используем bracket order или отдельные заявки
                        # В backtrader можно использовать sell с exectype=bt.Order.Stop для SL и bt.Order.Limit для TP
                        tp_order = self.sell(data=data, exectype=bt.Order.Limit, price=tp_price, size=size, parent=buy_order)
                        sl_order = self.sell(data=data, exectype=bt.Order.Stop, price=sl_price, size=size, parent=buy_order)
                        
                        self.orders[data._name] = buy_order
                        print(f"\t - Заявка на покупку создана: {buy_order.ref}")
                        print(f"\t - TP заявка: {tp_order.ref}, SL заявка: {sl_order.ref}")
                        print("-" * 50)

                else:  # Если позиция есть
                    if signal2 == 1:
                        # Принудительная продажа по сигналу (если не сработали TP/SL)
                        print("-" * 50)
                        print(f"\t - Продаем по рынку {data._name} (сигнал crossdown)...")
                        self.orders[data._name] = self.close()
                        print("-" * 50)

    def notify_order(self, order):
        """Изменение статуса заявки"""
        order_data_name = order.data._name
        print("*" * 50)
        self.log(f'Заявка номер {order.ref} {order.info.get("order_number", "")} {order.getstatusname()} {"Покупка" if order.isbuy() else "Продажа"} {order_data_name} {order.size} @ {order.price}')
        
        if order.status == bt.Order.Completed:
            if order.isbuy():
                self.log(f'Покупка {order_data_name} Цена: {order.executed.price:.2f}, Объём: {order.executed.value:.2f}, Комиссия: {order.executed.comm:.2f}')
            else:
                self.log(f'Продажа {order_data_name} Цена: {order.executed.price:.2f}, Объём: {order.executed.value:.2f}, Комиссия: {order.executed.comm:.2f}')
                # Сбрасываем заявку только если это была продажа по сигналу, а не TP/SL
                # Для TP/SL родительская заявка уже обработана
                if order.info.get('parent') is None:
                    self.orders[order_data_name] = None
        print("*" * 50)

    def notify_trade(self, trade):
        """Изменение статуса позиции"""
        if trade.isclosed:
            self.log(f'Прибыль по закрытой позиции {trade.getdataname()} Общая={trade.pnl:.2f}, Без комиссии={trade.pnlcomm:.2f}')
            # Дополнительная информация о причине закрытия
            if trade.pnlcomm > 0:
                reason = "Take Profit" if trade.pnlcomm >= (trade.size * trade.price * self.p.take_profit_pct / 100 * 0.9) else "Signal/Other"
                self.log(f'-> Закрытие вероятно по: {reason}')
            else:
                self.log(f'-> Закрытие по: Stop Loss или Signal')

    def notify_data(self, data, status, *args, **kwargs):
        """Изменение статуса приходящих баров"""
        data_status = data._getstatusname(status)
        _name = data._name if data._name else f"{self.p.name}"
        print(f'{_name} - {bt.TimeFrame.Names[data.p.timeframe]} {data.p.compression} - {data_status}')
        self.isLive = data_status == 'LIVE'
