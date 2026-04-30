import os
import moexalgo
import vectorbt as vbt
import pandas as pd
import numpy as np
import warnings
import traceback
from itertools import product

warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

print(f"📦 vectorbt версия: {vbt.__version__}")

# 🔍 НАСТРОЙКИ
DEBUG = True
CACHE_DIR = 'cache_data'
os.makedirs(CACHE_DIR, exist_ok=True)

INIT_CASH = 1_000_000
FEES = 0.001
MIN_TRADES = 1

# 📅 ПЕРИОД (ИСТОРИЧЕСКИЕ ДАННЫЕ вместо будущего)
START_DATE = pd.Timestamp('2023-01-01')
END_DATE = pd.Timestamp('2024-12-31')

results = []


# ──────────────────────────────────────────────
# 📥 ЗАГРУЗКА ДАННЫХ (МУЛЬТИ-ТФ + ЧАНКИ)
# ──────────────────────────────────────────────
def load_data_tf(ticker_name, start, end):
    """Загружает данные частями (чанками), чтобы обойти лимиты API"""
    tf_map = {'1D': '1d', '15T': '15min', '1T': '1min'}

    # Разбиваем период на чанки по 30 дней для надежной загрузки
    date_ranges = pd.date_range(start=start, end=end, freq='30D')
    ranges = []
    for i in range(len(date_ranges) - 1):
        ranges.append((date_ranges[i], date_ranges[i + 1]))
    # Добавляем последний хвост до end_date
    if date_ranges[-1] < end:
        ranges.append((date_ranges[-1], end))

    dataframes = {}

    for tf_name, tf_code in tf_map.items():
        cache_file = os.path.join(CACHE_DIR,
                                  f"{ticker_name}_{tf_name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")

        if os.path.exists(cache_file):
            print(f"📂 [{tf_name}] Загружаем из кэша...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                dataframes[tf_name] = df
                continue

        print(f"📥 [{tf_name}] Загружаем {ticker_name} с MOEX ({tf_code}) чанками...")
        ticker = moexalgo.Ticker(ticker_name)

        chunk_dfs = []

        for chunk_start, chunk_end in ranges:
            try:
                df_chunk = ticker.candles(
                    start=chunk_start.strftime('%Y-%m-%d'),
                    end=chunk_end.strftime('%Y-%m-%d'),
                    period=tf_code
                )

                if df_chunk is not None and not df_chunk.empty:
                    # Нормализация колонок
                    if 'date' in df_chunk.columns:
                        df_chunk = df_chunk.set_index('date')
                    elif 'begin' in df_chunk.columns:
                        df_chunk = df_chunk.set_index('begin')

                    df_chunk = df_chunk.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    })
                    df_chunk.index = pd.to_datetime(df_chunk.index)
                    chunk_dfs.append(df_chunk)

            except Exception as e:
                print(f"⚠️ Ошибка загрузки чанка [{tf_name}] {chunk_start}: {e}")

        if not chunk_dfs:
            print(f"❌ [{tf_name}] Не удалось получить данные.")
            return None

        # Объединение чанков
        df = pd.concat(chunk_dfs)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]  # Удаление дублей на стыках

        # Сохранение в кэш
        df.to_csv(cache_file)
        print(f"💾 [{tf_name}] Сохранено в кэш: {len(df)} баров (всего)")

        # Очистка данных
        needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[needed].dropna()
        for col in ['Close', 'High', 'Low']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(lower=0.01)

        dataframes[tf_name] = df

    return dataframes


# ──────────────────────────────────────────────
# 🧮 ИНДИКАТОРЫ
# ──────────────────────────────────────────────
def calculate_indicators(df, tf_name):
    c = df['Close']
    h = df['High']
    l = df['Low']

    # EMA Trend
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema32 = c.ewm(span=32, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    # ATR
    tr = pd.concat([h - l, h - c.shift(), c.shift() - l], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    # Doji Body Ratio
    body = abs(c - df['Open'])
    range_hl = h - l
    doji_ratio = body / range_hl.replace(0, np.nan)

    # Медиана ATR
    if tf_name == '1D':
        atr_med = atr.rolling(2).median()
    elif tf_name == '15T':
        atr_med = atr.rolling(20).median()
    else:  # 1T
        atr_med = atr.rolling(60).median()

    return pd.DataFrame({
        'ema10': ema10,
        'ema32': ema32,
        'rsi': rsi,
        'atr': atr,
        'atr_med': atr_med,
        'doji_ratio': doji_ratio,
        'close': c,
        'high': h,
        'low': l
    }).ffill().bfill()


# ──────────────────────────────────────────────
# 📤 ГЕНЕРАЦИЯ ОРДЕРОВ (МУЛЬТИ-ТФ ЛОГИКА)
# ──────────────────────────────────────────────
def generate_orders_multi_tf(data_dict, ind_dict, params, debug_stats=False):
    df_base = data_dict['1T']
    ind_base = ind_dict['1T']
    ind_15 = ind_dict['15T']
    ind_1d = ind_dict['1D']

    n = len(df_base)

    ts_15 = ind_15.reindex(df_base.index, method='ffill')
    ts_1d = ind_1d.reindex(df_base.index, method='ffill')

    directions = np.full(n, 1, dtype=np.int32)
    sizes = np.full(n, np.nan, dtype=np.float64)

    pos = 0
    entry_price = np.nan
    sl_price = np.nan
    tp_price = np.nan
    atr_entry = np.nan

    p = params

    # Статистика по фильтрам для диагностики
    stats = {
        'total_bars': n,
        'after_trend_filter': 0,
        'filtered_trend': 0,
        'after_vol_filter': 0,
        'filtered_vol_1d': 0,
        'filtered_vol_15': 0,
        'filtered_vol_1m': 0,
        'after_doji_filter': 0,
        'filtered_doji': 0,
        'after_rsi_filter': 0,
        'filtered_rsi': 0,
        'entries': 0,
        'exits_sl': 0,
        'exits_tp': 0
    }

    for i in range(n):
        close_i = ind_base['close'].iloc[i]
        high_i = ind_base['high'].iloc[i]
        low_i = ind_base['low'].iloc[i]
        atr_i = ind_base['atr'].iloc[i]

        rsi_1d = ts_1d['rsi'].iloc[i]
        rsi_15 = ts_15['rsi'].iloc[i]
        rsi_1m = ind_base['rsi'].iloc[i]

        atr_1d = ts_1d['atr'].iloc[i]
        atr_1d_med = ts_1d['atr_med'].iloc[i]

        atr_15 = ts_15['atr'].iloc[i]
        atr_15_med = ts_15['atr_med'].iloc[i]

        atr_1m = ind_base['atr'].iloc[i]
        atr_1m_med = ind_base['atr_med'].iloc[i]

        doji_1m = ind_base['doji_ratio'].iloc[i]

        # 1. ФИЛЬТР ТРЕНДА (1D)
        trend_1d = (ts_1d['close'].iloc[i] > ts_1d['ema10'].iloc[i]) and \
                   (ts_1d['ema10'].iloc[i] > ts_1d['ema32'].iloc[i])

        if not trend_1d:
            stats['filtered_trend'] += 1
            continue
        stats['after_trend_filter'] += 1

        # 2. ФИЛЬТР ВОЛАТИЛЬНОСТИ
        vol_pass_1d = False
        if not pd.isna(atr_1d) and not pd.isna(atr_1d_med):
            if atr_1d > p['vol_1d_mult'] * atr_1d_med:
                vol_pass_1d = True

        vol_pass_15 = False
        if not pd.isna(atr_15) and not pd.isna(atr_15_med):
            if atr_15 > p['vol_15_mult'] * atr_15_med:
                vol_pass_15 = True

        vol_pass_1m = False
        if not pd.isna(atr_1m) and not pd.isna(atr_1m_med):
            if atr_1m > p['vol_1m_mult'] * atr_1m_med:
                vol_pass_1m = True

        if not vol_pass_1d:
            stats['filtered_vol_1d'] += 1
            continue
        if not vol_pass_15:
            stats['filtered_vol_15'] += 1
            continue
        if not vol_pass_1m:
            stats['filtered_vol_1m'] += 1
            continue
        stats['after_vol_filter'] += 1

        # 3. Doji Filter
        if not pd.isna(doji_1m):
            if doji_1m < 0.3:
                stats['filtered_doji'] += 1
                continue

        stats['after_doji_filter'] += 1

        # 4. ФИЛЬТР RSI
        rsi_pass = (rsi_1d < p['rsi_1d_thresh']) and \
                   (rsi_15 < p['rsi_15_thresh']) and \
                   (rsi_1m < p['rsi_1m_thresh'])

        if not rsi_pass:
            stats['filtered_rsi'] += 1
            continue

        stats['after_rsi_filter'] += 1

        # === СИГНАЛ НА ВХОД ===
        if pos == 0:
            pos = 1
            entry_price = close_i
            atr_entry = atr_i
            sl_price = entry_price - p['sl_atr_mult'] * atr_entry
            tp_price = entry_price + p['tp_atr_mult'] * atr_entry

            directions[i] = 1
            sizes[i] = 1.0
            stats['entries'] += 1

        # === УПРАВЛЕНИЕ ПОЗИЦИЕЙ ===
        elif pos == 1:
            if low_i <= sl_price:
                directions[i] = 2
                sizes[i] = 1.0
                pos = 0
                stats['exits_sl'] += 1
            elif high_i >= tp_price:
                directions[i] = 2
                sizes[i] = 1.0
                pos = 0
                stats['exits_tp'] += 1

    if debug_stats:
        print(f"\n📊 СТАТИСТИКА ФИЛЬТРОВ для параметров: {p}")
        print(f"   Всего баров: {stats['total_bars']}")
        print(f"   После тренд-фильтра: {stats['after_trend_filter']} (отфильтровано: {stats['filtered_trend']})")
        print(f"   После волатильности: {stats['after_vol_filter']}")
        print(f"      ├─ Отфильтровано 1D: {stats['filtered_vol_1d']}")
        print(f"      ├─ Отфильтровано 15T: {stats['filtered_vol_15']}")
        print(f"      └─ Отфильтровано 1M: {stats['filtered_vol_1m']}")
        print(f"   После Doji-фильтра: {stats['after_doji_filter']} (отфильтровано: {stats['filtered_doji']})")
        print(f"   После RSI-фильтра: {stats['after_rsi_filter']} (отфильтровано: {stats['filtered_rsi']})")
        print(f"   Итого сигналов на вход: {stats['entries']}")
        print(f"   Выходы по SL: {stats['exits_sl']}, по TP: {stats['exits_tp']}")
        print("-" * 60)

    return directions, sizes


# ──────────────────────────────────────────────
# 🔍 ОПТИМИЗАЦИЯ
# ──────────────────────────────────────────────

print("🚀 Старт загрузки данных...")
data_dict = load_data_tf('SBER', START_DATE, END_DATE)

if data_dict is None or any(df.empty for df in data_dict.values()):
    print("❌ Критическая ошибка: Не удалось загрузить данные.")
    exit(1)

print("🧮 Расчет индикаторов...")
ind_dict = {}
for tf, df in data_dict.items():
    ind_dict[tf] = calculate_indicators(df, tf)

# Сетка параметров (ИСПРАВЛЕННЫЙ СИНТАКСИС)
param_grid = {
    'vol_1d_mult': [0.5, 0.8],
    'vol_15_mult': [0.5, 0.8],
    'vol_1m_mult': [1.0, 1.5, 2.0],

    'rsi_1d_thresh': [40, 35, 30],
    'rsi_15_thresh': [90, 35, 40],
    'rsi_1m_thresh': [25, 30, 35],

    'sl_atr_mult': [1.5],
    'tp_atr_mult': [1.5],

    'doji_thresh': [0.1]
}

best_sharpe = -999
best_params = None
error_count = 0
cfg_count = 0

print("🔄 Запуск оптимизации...")
price_arr = np.ascontiguousarray(ind_dict['1T']['close'].values.astype(np.float64))

# Для отладки можно включить статистику по фильтрам (первый прогон или лучший конфиг)
DEBUG_FILTERS = True  # Включить вывод статистики фильтров

for vals in product(*param_grid.values()):
    p = dict(zip(param_grid.keys(), vals))
    cfg_count += 1

    try:
        # Показываем детальную статистику фильтров для первых нескольких конфигов
        show_stats = DEBUG_FILTERS and cfg_count <= 5
        dirs, sizes = generate_orders_multi_tf(data_dict, ind_dict, p, debug_stats=show_stats)

        active_mask = ~np.isnan(sizes)
        if active_mask.sum() == 0:
            if DEBUG: print(f"⏭️ Конфиг #{cfg_count}: Нет сделок")
            continue

        dirs_arr = np.ascontiguousarray(dirs)
        sizes_arr = np.ascontiguousarray(sizes)

        pf = vbt.Portfolio.from_orders(
            close=price_arr,
            direction=dirs_arr,
            size=sizes_arr,
            size_type='percent',
            init_cash=float(INIT_CASH),
            fees=FEES,
            allow_partial=True,
            cash_sharing=False,
            lock_cash=False
        )

        t = int(pf.trades.count())
        if t < MIN_TRADES:
            continue

        sr = float(pf.sharpe_ratio())
        dd = float(pf.max_drawdown())
        ret = float(pf.total_return())
        wr = float(pf.trades.win_rate()) * 100

        if pd.isna(sr) or pd.isinf(sr) or dd > 0.5:
            continue

        if DEBUG:
            print(f"✅ Конфиг #{cfg_count} | Сделок: {t} | SR: {sr:.2f} | Ret: {ret:.2%}")

        if sr > best_sharpe:
            best_sharpe = sr
            best_params = p
            # Показываем полную статистику фильтров для нового лидера
            dirs_debug, sizes_debug = generate_orders_multi_tf(data_dict, ind_dict, p, debug_stats=True)
            
            results.append({
                'Sharpe': round(sr, 2),
                'Return': round(ret * 100, 2),
                'DD': round(dd * 100, 2),
                'Trades': t,
                'WinRate': round(wr, 1),
                'Params': p
            })
            print(f"🏆 Новый лидер! Sharpe: {sr:.2f}")

    except Exception as e:
        error_count += 1
        print(f"❌ Ошибка #{cfg_count}: {e}")
        if DEBUG: traceback.print_exc()

# ──────────────────────────────────────────────
# 📊 ИТОГИ
# ──────────────────────────────────────────────
if results:
    df_res = pd.DataFrame(results).sort_values('Sharpe', ascending=False).reset_index(drop=True)
    print("\n" + "=" * 30)
    print("🏆 ТОП РЕЗУЛЬТАТЫ")
    print("=" * 30)
    print(df_res.head(5).to_string(index=False))

    best = df_res.iloc[0]
    print(f"\n📌 Лучшие параметры: {best['Params']}")

    df_res.drop(columns='Params').to_csv('sngs_multi_tf_optimization.csv', index=False, encoding='utf-8-sig')
    print("💾 Отчет сохранен: sngs_multi_tf_optimization.csv")
else:
    print("\n⚠️ Стратегия не совершила ни одной успешной сделки или не прошла фильтры.")
    print("Возможные причины:")
    print("1. Условия входа слишком строгие.")
    print("2. Тренд 1D (EMA10 > EMA32) отсутствовал в выбранные даты.")
