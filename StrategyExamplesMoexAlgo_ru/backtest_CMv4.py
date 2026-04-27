import backtrader as bt
import datetime
import sys
import os
import pickle
import hashlib

from strategy_cmv4_pl import StrategyCMV4PL
from backtrader_moexalgo.moexalgo_store import MoexAlgoStore


# Директория для кэширования данных
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')


def get_cache_key(symbol, fromdate, todate, timeframe, compression, metric):
    """Генерация уникального ключа для кэша на основе параметров"""
    key_str = f"{symbol}_{fromdate}_{todate}_{timeframe}_{compression}_{metric}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_from_cache(cache_key):
    """Загрузка данных из кэша"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Данные загружены из кэша: {cache_file}")
            return data
        except Exception as e:
            print(f"⚠ Ошибка загрузки из кэша: {e}")
    return None


def save_to_cache(cache_key, data):
    """Сохранение данных в кэш"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Данные сохранены в кэш: {cache_file}")
    except Exception as e:
        print(f"⚠ Ошибка сохранения в кэш: {e}")


def run_daily_backtest(date_start, date_end, symbol='SNGS', use_cache=True, force_update=False, try_super_candles=True):
    """Запуск бэктеста для одного дня"""
    cerebro = bt.Cerebro()
    
    # Добавляем стратегию CMV4
    cerebro.addstrategy(StrategyCMV4PL)
    
    # Параметры для кэширования
    timeframe = bt.TimeFrame.Minutes
    compression = 10
    metric = 'tradestats' if try_super_candles else None  # Пытаемся использовать tradestats
    
    cache_key = get_cache_key(symbol, date_start, date_end, timeframe, compression, metric)
    
    # Пытаемся загрузить из кэша
    data = None
    data_points = None
    if use_cache and not force_update:
        loaded_data = load_from_cache(cache_key)
        if loaded_data is not None:
            # Если данные загружены из кэша и они пустые - возвращаем результат
            if isinstance(loaded_data, list) and len(loaded_data) == 0:
                print(f"⚠ Нет данных в кэше для {symbol} за {date_start.date()}")
                return {
                    'date': date_start.date(),
                    'start_cash': 100000.0,
                    'end_cash': 100000.0,
                    'pnl': 0.0,
                    'pnl_percent': 0.0,
                    'trades_count': 0,
                    'strategy': None,
                    'error': 'no_data'
                }
            # Данные успешно загружены из кэша
            data_points = loaded_data
    
    # Если нет в кэше или требуется обновление - загружаем через moexalgo
    if data_points is None:
        print(f"📡 Загрузка данных для {symbol} ({date_start.date()} - {date_end.date()})...")
        store = MoexAlgoStore()
        
        # Сначала пытаемся загрузить с super_candles (если try_super_candles=True)
        use_super_candles = try_super_candles
        
        while True:
            # Параметры для запроса
            getdata_kwargs = {
                'dataname': symbol,
                'fromdate': date_start,
                'todate': date_end,
                'timeframe': timeframe,
                'compression': compression,
                'live_bars': False
            }
            
            # Добавляем super_candles и metric только если используем их
            if use_super_candles:
                getdata_kwargs['super_candles'] = True
                getdata_kwargs['metric'] = 'tradestats'
            
            try:
                data = store.getdata(**getdata_kwargs)
                
                # Проверяем, есть ли данные (итерируем для загрузки)
                data_points = []
                has_data = False
                
                for d in data:
                    # Проверка на наличие данных (защита от пустых баров)
                    if len(d.open) == 0 or len(d.high) == 0 or len(d.low) == 0 or len(d.close) == 0:
                        continue
                    has_data = True
                    point = {
                        'datetime': d.datetime[0],
                        'open': d.open[0],
                        'high': d.high[0],
                        'low': d.low[0],
                        'close': d.close[0],
                        'volume': d.volume[0]
                    }
                    data_points.append(point)
                
                # Если данных нет вообще - сохраняем пустой кэш и выходим
                if not has_data:
                    print(f"⚠ Нет данных для {symbol} за {date_start.date()}")
                    if use_cache:
                        save_to_cache(cache_key, [])
                    return {
                        'date': date_start.date(),
                        'start_cash': 100000.0,
                        'end_cash': 100000.0,
                        'pnl': 0.0,
                        'pnl_percent': 0.0,
                        'trades_count': 0,
                        'strategy': None,
                        'error': 'no_data'
                    }
                
                # Если использовали super_candles, пробуем добавить tradestats
                if use_super_candles:
                    if hasattr(data, 'p') and hasattr(data.p, 'supercandles') and symbol in data.p.supercandles:
                        sc_list = data.p.supercandles[symbol].get('tradestats', [])
                        # Переворачиваем список, чтобы привести к правильному порядку
                        sc_list_reversed = list(reversed(sc_list))
                        
                        # Добавляем поля tradestats к соответствующим точкам данных
                        for i, point in enumerate(data_points):
                            if i < len(sc_list_reversed):
                                sc_data = sc_list_reversed[i]
                                if sc_data:
                                    for key, value in sc_data.items():
                                        point[f'tradestats_{key}'] = value
                        print("✓ Расширенные данные (tradestats) успешно загружены")
                    else:
                        # Super candles не доступны (нет подписки), пробуем без них
                        print("⚠ Расширенные данные (tradestats) недоступны. Требуется платная подписка MOEX AlgoPack.")
                        print("  Продолжение работы с обычными свечами...")
                        use_super_candles = False
                        continue  # Повторяем запрос без super_candles
                
                # Сохраняем в кэш
                if use_cache:
                    save_to_cache(cache_key, data_points)
                
                break  # Выход из цикла while после успешной загрузки
                
            except Exception as e:
                error_msg = str(e)
                if use_super_candles and ('403' in error_msg or 'Forbidden' in error_msg or 'tradestats' in error_msg.lower()):
                    # Ошибка доступа к tradestats - пробуем без них
                    print(f"⚠ Ошибка доступа к расширенным данным: {e}")
                    print("  Продолжение работы с обычными свечами...")
                    use_super_candles = False
                    continue  # Повторяем запрос без super_candles
                else:
                    # Другая ошибка - пробрасываем дальше
                    raise
    
    # Создаем объект data из data_points (из кэша или загруженных)
    import pandas as pd
    data = bt.feeds.PandasData(dataname=pd.DataFrame(data_points))
    data._dataname = symbol
    
    cerebro.adddata(data)
    
    # Параметры брокера
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0005)  # 0.05% комиссия (5 bps из cost_model)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)  # Размер позиции
    
    # Запуск
    results = cerebro.run()
    strat = results[0] if results else None
    
    # Получаем статистику
    final_cash = cerebro.broker.getvalue()
    pnl = final_cash - 100000.0
    pnl_percent = (pnl / 100000.0) * 100
    
    # Считаем количество сделок через analyzer или trades observer
    trades_count = 0
    if strat:
        trades_count = len([t for t in strat._trades]) if hasattr(strat, '_trades') else 0
        if trades_count == 0:
            # Альтернативный способ - через observers
            trades_count = len(cerebro.observer.trades) if hasattr(cerebro, 'observer') and hasattr(cerebro.observer, 'trades') else 0
    
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
    print("БЭКТЕСТ СТРАТЕГИИ SNGS Channel Macro v4 (365 дней)")
    print("=" * 60)
    
    # Определяем диапазон дат (последние 365 календарных дней)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    
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
                
                # Пропускаем дни без данных (не добавляем в статистику)
                if result.get('error') == 'no_data':
                    continue
                
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
        print("\n=== ИТОГОВАЯ СТАТИСТИКА ЗА 365 ДНЕЙ ===")
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
        
        # Расчет Sharpe ratio (упрощенно, на основе дневных доходностей)
        if len(daily_results) > 1:
            daily_returns = [r['pnl_percent'] for r in daily_results]
            import statistics
            avg_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
            print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
            
            # Profit Factor
            gross_profit = sum(r['pnl'] for r in daily_results if r['pnl'] > 0)
            gross_loss = abs(sum(r['pnl'] for r in daily_results if r['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            print(f"Profit Factor: {profit_factor:.2f}")
            
            # Win Rate
            win_rate = winning_days / len(daily_results) * 100
            print(f"Win Rate: {win_rate:.1f}%")
    else:
        print("\n=== НЕТ ДАННЫХ ДЛЯ АНАЛИЗА ===")
        print("Возможно, указаны будущие даты или нет доступа к API MOEX")
    print("=" * 60)


if __name__ == '__main__':
    main()
