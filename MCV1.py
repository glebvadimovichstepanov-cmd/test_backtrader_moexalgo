"C:\Users\Gleb Stepanov\AppData\Local\Programs\Python\Python314\python.exe" "C:\Users\Gleb Stepanov\PycharmProjects\test_backtrader_moexalgo\StrategyExamplesMoexAlgo_ru\CMV1.py" 
📦 vectorbt версия: 1.0.0
📂 Загружаем данные из кэша...
✅ Данные: 26519 баров | 2025-06-01 09:55:00 → 2026-04-14 18:35:00
📊 Цена: 19.69 - 25.02
⚙️ Расчёт индикаторов и тренда старшего ТФ...
🔄 Запуск сеточной оптимизации...

🔍 Конфиг #1: {'confluence_thresh': 2, 'vol_ratio_thresh': 1.0, 'sl_atr_mult': 1.0, 'tp_atr_mult': 2.0, 'avg_atr_mult': 1.0, 'avg_size': 0.5}
  📊 direction unique: [-1  1] | active orders: 4946

❌ Ошибка в конфиге #1 {'confluence_thresh': 2, 'vol_ratio_thresh': 1.0, 'sl_atr_mult': 1.0, 'tp_atr_mult': 2.0, 'avg_atr_mult': 1.0, 'avg_size': 0.5}:

⚠️ Не найдено конфигураций. (Всего ошибок: 1)
Traceback (most recent call last):
  File "C:\Users\Gleb Stepanov\PycharmProjects\test_backtrader_moexalgo\StrategyExamplesMoexAlgo_ru\CMV1.py", line 301, in <module>
    pf = vbt.Portfolio.from_orders(
        close=close_arr,
    ...<8 lines>...
        lock_cash=False
    )
  File "C:\Users\Gleb Stepanov\PycharmProjects\test-vectorbt\vectorbt\portfolio\base.py", line 2012, in from_orders
    order_records, log_records = dispatch.simulate_from_orders(
                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        target_shape_2d,
        ^^^^^^^^^^^^^^^^
    ...<10 lines>...
        engine=engine,
        ^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\Gleb Stepanov\PycharmProjects\test-vectorbt\vectorbt\portfolio\dispatch.py", line 703, in simulate_from_orders
    return simulate_from_orders_nb(
        target_shape,
    ...<25 lines>...
        flex_2d=flex_2d,
    )
  File "C:\Users\Gleb Stepanov\PycharmProjects\test-vectorbt\vectorbt\portfolio\nb.py", line 399, in execute_order_nb
    raise ValueError("order.direction is invalid")
ValueError: order.direction is invalid

Process finished with exit code 0
