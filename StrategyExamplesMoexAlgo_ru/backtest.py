import backtrader as bt
import datetime
import sys
import os

from strategy_mc_pl import StrategyMCWithTPSL
from backtrader_moexalgo.moexalgo_store import MoexAlgoStore

def run_daily_backtest(date_start, date_end, symbol='SNGS'):
    """Запуск бэктеста для одного дня"""
    cerebro = bt.Cerebro()
    
    # Добавляем стратегию
    cerebro.addstrategy(StrategyMCWithTPSL)
    
    # Создаем хранилище и получаем данные
    store = MoexAlgoStore()
    data = store.getdata(
        dataname=symbol,
        fromdate=date_start,
        todate=date_end,
        timeframe=bt.TimeFrame.Minutes,
        compression=5,
        metric='tradestats',  # Используем расширенные данные для интрадея
        live_bars=False
    )
    cerebro.adddata(data)
    
    # Параметры брокера
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0004)  # 0.04% комиссия
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)  # Размер позиции
    
    # Запуск
    results = cerebro.run()
    strat = results[0]
    
    # Получаем статистику
    final_cash = cerebro.broker.getvalue()
    pnl = final_cash - 100000.0
    pnl_percent = (pnl / 100000.0) * 100
    
    # Считаем количество сделок
    trades_count = len(strat.trades)
    
    return {
        'date': date_start.date(),
        'start_cash': 100000.0,
        'end_cash': final_cash,
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'trades_count': trades_count,
        'strategy': strat
    }

def main():
    print("=" * 60)
    print("БЭКТЕСТ СТРАТЕГИИ StrategyMCWithTPSL (30 дней)")
    print("=" * 60)
    
    # Определяем диапазон дат (последние 30 календарных дней)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    
    # Для интрадея лучше тестировать по отдельным дням
    # Генерируем список торговых дней (упрощенно - каждый день)
    current_date = start_date
    daily_results = []
    total_pnl = 0.0
    total_trades = 0
    winning_days = 0
    losing_days = 0
    
    print(f"\nПериод тестирования: {start_date.date()} - {end_date.date()}")
    print("-" * 60)
    print(f"{'Дата':<12} | {'PnL (руб)':<12} | {'PnL (%)':<10} | {'Сделки':<8}")
    print("-" * 60)
    
    while current_date <= end_date:
        # Пропускаем выходные (суббота=5, воскресенье=6)
        if current_date.weekday() < 5:
            try:
                # Устанавливаем начало и конец дня для интрадей теста
                day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = current_date.replace(hour=23, minute=59, second=59, microsecond=0)
                
                result = run_daily_backtest(day_start, day_end)
                daily_results.append(result)
                
                total_pnl += result['pnl']
                total_trades += result['trades_count']
                
                if result['pnl'] > 0:
                    winning_days += 1
                elif result['pnl'] < 0:
                    losing_days += 1
                
                # Вывод статистики за день
                color = "\033[92m" if result['pnl'] >= 0 else "\033[91m"  # Зеленый/Красный
                reset = "\033[0m"
                print(f"{color}{result['date']:<12} | {result['pnl']:>10.2f}  | {result['pnl_percent']:>8.2f}% | {result['trades_count']:>8}{reset}")
                
            except Exception as e:
                print(f"{current_date.date()} | Ошибка: {str(e)[:30]}...")
        
        current_date += datetime.timedelta(days=1)
    
    # Итоговая статистика
    print("-" * 60)
    if len(daily_results) > 0:
        print("\n=== ИТОГОВАЯ СТАТИСТИКА ЗА 30 ДНЕЙ ===")
        print(f"Общий PnL: {total_pnl:.2f} руб.")
        print(f"Общий доход (%): {(total_pnl / (100000.0 * len(daily_results))) * 100:.2f}% (от среднего дневного депозита)")
        print(f"Всего сделок: {total_trades}")
        print(f"Торговых дней: {len(daily_results)}")
        print(f"Прибыльных дней: {winning_days} ({(winning_days/len(daily_results)*100):.1f}%)")
        print(f"Убыточных дней: {losing_days} ({(losing_days/len(daily_results)*100):.1f}%)")
        
        if total_trades > 0:
            avg_trade_pnl = total_pnl / total_trades
            print(f"Средний PnL на сделку: {avg_trade_pnl:.2f} руб.")
        
        # Расчет максимального просадки (упрощенно)
        peak = 100000.0
        max_drawdown = 0.0
        current_cash = 100000.0
        
        for res in daily_results:
            current_cash += res['pnl']
            if current_cash > peak:
                peak = current_cash
            drawdown = (peak - current_cash) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        print(f"Максимальная просадка: {max_drawdown:.2f}%")
    else:
        print("\n=== НЕТ ДАННЫХ ДЛЯ АНАЛИЗА ===")
        print("Возможно, указаны будущие даты или нет доступа к API MOEX")
    print("=" * 60)

if __name__ == '__main__':
    main()
