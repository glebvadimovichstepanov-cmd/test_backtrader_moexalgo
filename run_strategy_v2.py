import os
import moexalgo
import vectorbt as vbt
import pandas as pd
import numpy as np
import warnings
import traceback
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

# Массив цен для backtesting (используем Close с 1минутного таймфрейма)
price_arr = data_dict['1T']['Close'].values

# Базовая сетка параметров для начального поиска
param_grid_base = {
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

def evaluate_params(p, data_dict, ind_dict, price_arr, cfg_count_ref):
    """Вычисляет метрики для заданных параметров"""
    try:
        dirs, sizes = generate_orders_multi_tf(data_dict, ind_dict, p, debug_stats=False)
        
        active_mask = ~np.isnan(sizes)
        if active_mask.sum() == 0:
            return None, 0
        
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
            return None, 0
        
        sr = float(pf.sharpe_ratio())
        dd = float(pf.max_drawdown())
        ret = float(pf.total_return())
        wr = float(pf.trades.win_rate()) * 100
        
        if pd.isna(sr) or pd.isinf(sr) or dd > 0.5:
            return None, 0
        
        return {
            'sharpe': sr,
            'return': ret,
            'dd': dd,
            'trades': t,
            'win_rate': wr,
            'params': p.copy()
        }, t
        
    except Exception as e:
        return None, 0


def evaluate_params_mp(args):
    """Обертка для многопроцессорности (сериализуемая версия)"""
    p, data_dict_serialized, ind_dict_serialized, price_arr, cfg_count_ref = args
    
    # Восстанавливаем данные из сериализованного формата
    import pickle
    data_dict = pickle.loads(data_dict_serialized)
    ind_dict = pickle.loads(ind_dict_serialized)
    
    return evaluate_params(p, data_dict, ind_dict, price_arr, cfg_count_ref)


def random_search_optimization(data_dict, ind_dict, price_arr, param_ranges, n_iterations=100):
    """Случайный поиск оптимальных параметров с многопроцессорностью"""
    best_result = None
    best_sharpe = -999
    
    print(f"\n🔎 Случайный поиск ({n_iterations} итераций, процессов: {multiprocessing.cpu_count()})...")
    
    # Сериализуем данные для передачи в процессы
    import pickle
    data_dict_serialized = pickle.dumps(data_dict)
    ind_dict_serialized = pickle.dumps(ind_dict)
    
    # Генерируем все параметры заранее
    params_list = []
    for i in range(n_iterations):
        p = {
            'vol_1d_mult': np.random.uniform(param_ranges['vol_1d_mult'][0], param_ranges['vol_1d_mult'][1]),
            'vol_15_mult': np.random.uniform(param_ranges['vol_15_mult'][0], param_ranges['vol_15_mult'][1]),
            'vol_1m_mult': np.random.uniform(param_ranges['vol_1m_mult'][0], param_ranges['vol_1m_mult'][1]),
            'rsi_1d_thresh': np.random.uniform(param_ranges['rsi_1d_thresh'][0], param_ranges['rsi_1d_thresh'][1]),
            'rsi_15_thresh': np.random.uniform(param_ranges['rsi_15_thresh'][0], param_ranges['rsi_15_thresh'][1]),
            'rsi_1m_thresh': np.random.uniform(param_ranges['rsi_1m_thresh'][0], param_ranges['rsi_1m_thresh'][1]),
            'sl_atr_mult': np.random.uniform(param_ranges['sl_atr_mult'][0], param_ranges['sl_atr_mult'][1]),
            'tp_atr_mult': np.random.uniform(param_ranges['tp_atr_mult'][0], param_ranges['tp_atr_mult'][1]),
            'doji_thresh': np.random.uniform(param_ranges['doji_thresh'][0], param_ranges['doji_thresh'][1])
        }
        params_list.append((p, data_dict_serialized, ind_dict_serialized, price_arr, i))
    
    # Запускаем многопроцессорную обработку
    max_workers = min(multiprocessing.cpu_count(), n_iterations)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_params_mp, args) for args in params_list]
        
        completed = 0
        for future in as_completed(futures):
            result, trades = future.result()
            completed += 1
            
            if result is not None and result['sharpe'] > best_sharpe:
                best_sharpe = result['sharpe']
                best_result = result
                if completed % 20 == 0:
                    print(f"  Обработано {completed}/{n_iterations}: текущий Sharpe={result['sharpe']:.2f}, Trades={result['trades']}")
    
    return best_result


def refine_optimization(data_dict, ind_dict, price_arr, best_params, param_ranges, n_iterations=50):
    """Уточняющая оптимизация вокруг лучших параметров с многопроцессорностью"""
    best_result = {'sharpe': -999, 'params': best_params}
    sharpe_baseline = best_result['sharpe']
    
    # Вычисляем базовый Sharpe для лучших параметров
    base_result, _ = evaluate_params(best_params, data_dict, ind_dict, price_arr, 0)
    if base_result:
        best_result = base_result
        sharpe_baseline = base_result['sharpe']
    
    print(f"\n🔬 Уточняющая оптимизация вокруг лучших параметров ({n_iterations} итераций, процессов: {multiprocessing.cpu_count()})...")
    print(f"   Базовый Sharpe: {sharpe_baseline:.2f}")
    
    # Сериализуем данные для передачи в процессы
    import pickle
    data_dict_serialized = pickle.dumps(data_dict)
    ind_dict_serialized = pickle.dumps(ind_dict)
    
    # Сужаем диапазоны вокруг лучших значений (10% от диапазона)
    narrow_ranges = {}
    for key in best_params.keys():
        center = best_params[key]
        range_size = param_ranges[key][1] - param_ranges[key][0]
        narrow_range = range_size * 0.1
        narrow_ranges[key] = [
            max(param_ranges[key][0], center - narrow_range),
            min(param_ranges[key][1], center + narrow_range)
        ]
    
    # Генерируем все параметры заранее
    params_list = []
    for i in range(n_iterations):
        p = {
            'vol_1d_mult': np.random.uniform(narrow_ranges['vol_1d_mult'][0], narrow_ranges['vol_1d_mult'][1]),
            'vol_15_mult': np.random.uniform(narrow_ranges['vol_15_mult'][0], narrow_ranges['vol_15_mult'][1]),
            'vol_1m_mult': np.random.uniform(narrow_ranges['vol_1m_mult'][0], narrow_ranges['vol_1m_mult'][1]),
            'rsi_1d_thresh': np.random.uniform(narrow_ranges['rsi_1d_thresh'][0], narrow_ranges['rsi_1d_thresh'][1]),
            'rsi_15_thresh': np.random.uniform(narrow_ranges['rsi_15_thresh'][0], narrow_ranges['rsi_15_thresh'][1]),
            'rsi_1m_thresh': np.random.uniform(narrow_ranges['rsi_1m_thresh'][0], narrow_ranges['rsi_1m_thresh'][1]),
            'sl_atr_mult': np.random.uniform(narrow_ranges['sl_atr_mult'][0], narrow_ranges['sl_atr_mult'][1]),
            'tp_atr_mult': np.random.uniform(narrow_ranges['tp_atr_mult'][0], narrow_ranges['tp_atr_mult'][1]),
            'doji_thresh': np.random.uniform(narrow_ranges['doji_thresh'][0], narrow_ranges['doji_thresh'][1])
        }
        params_list.append((p, data_dict_serialized, ind_dict_serialized, price_arr, i))
    
    # Запускаем многопроцессорную обработку
    max_workers = min(multiprocessing.cpu_count(), n_iterations)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_params_mp, args) for args in params_list]
        
        completed = 0
        for future in as_completed(futures):
            result, trades = future.result()
            completed += 1
            
            if result is not None and result['sharpe'] > best_result['sharpe']:
                improvement = result['sharpe'] - sharpe_baseline
                best_result = result
                if completed % 10 == 0:
                    print(f"  Обработано {completed}/{n_iterations}: Sharpe={result['sharpe']:.2f} (+{improvement:.2f}), Trades={result['trades']}")
    
    return best_result


def grid_search_optimization(data_dict, ind_dict, price_arr, param_grid, start_cfg=1, debug_filters=False):
    """Полный перебор по сетке параметров"""
    best_sharpe = -999
    best_params = None
    results = []
    error_count = 0
    cfg_count = start_cfg - 1
    
    DEBUG_FILTERS_FLAG = debug_filters
    
    for vals in product(*param_grid.values()):
        p = dict(zip(param_grid.keys(), vals))
        cfg_count += 1
        
        show_stats = DEBUG_FILTERS_FLAG and cfg_count <= 5
        
        result, trades = evaluate_params(p, data_dict, ind_dict, price_arr, cfg_count)
        
        if result is None:
            if DEBUG and cfg_count <= 10:
                print(f"⏭️ Конфиг #{cfg_count}: Нет сделок или не прошел фильтры")
            continue
        
        if DEBUG and cfg_count % 50 == 0:
            print(f"✅ Конфиг #{cfg_count} | Сделок: {result['trades']} | SR: {result['sharpe']:.2f} | Ret: {result['return']:.2%}")
        
        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_params = p
            
            # Показываем полную статистику фильтров для нового лидера
            if debug_filters:
                dirs_debug, sizes_debug = generate_orders_multi_tf(data_dict, ind_dict, p, debug_stats=True)
            
            results.append({
                'Sharpe': round(result['sharpe'], 2),
                'Return': round(result['return'] * 100, 2),
                'DD': round(result['dd'] * 100, 2),
                'Trades': result['trades'],
                'WinRate': round(result['win_rate'], 1),
                'Params': p.copy()
            })
            print(f"🏆 Новый лидер! Sharpe: {result['sharpe']:.2f}, Trades: {result['trades']}")
    
    return best_params, best_sharpe, results, cfg_count


# Определяем диапазоны параметров для автоматической оптимизации
param_ranges = {
    'vol_1d_mult': [0.3, 1.5],
    'vol_15_mult': [0.3, 1.5],
    'vol_1m_mult': [0.5, 3.0],
    'rsi_1d_thresh': [20, 50],
    'rsi_15_thresh': [20, 95],
    'rsi_1m_thresh': [15, 50],
    'sl_atr_mult': [1.0, 3.0],
    'tp_atr_mult': [1.0, 3.0],
    'doji_thresh': [0.05, 0.4]
}

best_sharpe_global = -999
best_params_global = None
all_results = []


def run_optimization():
    """Основная функция оптимизации"""
    global best_sharpe_global, best_params_global, all_results
    
    print("=" * 60)
    print("🚀 ЗАПУСК АВТОМАТИЧЕСКОГО ПОДБОРА ПАРАМЕТРОВ")
    print("=" * 60)
    
    # Этап 1: Грубый случайный поиск для исследования пространства
    print("\n📊 ЭТАП 1: Грубый случайный поиск (исследование пространства)")
    random_result = random_search_optimization(data_dict, ind_dict, price_arr, param_ranges, n_iterations=200)
    
    if random_result:
        best_sharpe_global = random_result['sharpe']
        best_params_global = random_result['params']
        all_results.append(random_result)
        print(f"\n✅ Лучший результат после случайного поиска:")
        print(f"   Sharpe: {best_sharpe_global:.2f}")
        print(f"   Параметры: {best_params_global}")
    else:
        print("⚠️ Случайный поиск не дал результатов, используем базовую сетку")
    
    # Этап 2: Уточняющая оптимизация вокруг лучших найденных параметров
    if best_params_global:
        print("\n📊 ЭТАП 2: Уточняющая оптимизация")
        refined_result = refine_optimization(data_dict, ind_dict, price_arr, best_params_global, param_ranges, n_iterations=100)
        
        if refined_result['sharpe'] > best_sharpe_global:
            best_sharpe_global = refined_result['sharpe']
            best_params_global = refined_result['params']
            all_results.append(refined_result)
            print(f"\n✅ Улучшено после уточнения:")
            print(f"   Sharpe: {best_sharpe_global:.2f}")
            print(f"   Параметры: {best_params_global}")

    # Этап 3: Финальная тонкая настройка (еще более узкий диапазон)
    if best_params_global:
        print("\n📊 ЭТАП 3: Финальная тонкая настройка")
        # Еще больше сужаем диапазоны (5% от исходного)
        fine_ranges = {}
        for key in best_params_global.keys():
            center = best_params_global[key]
            range_size = param_ranges[key][1] - param_ranges[key][0]
            fine_range = range_size * 0.05
            fine_ranges[key] = [
                max(param_ranges[key][0], center - fine_range),
                min(param_ranges[key][1], center + fine_range)
            ]
        
        fine_result = refine_optimization(data_dict, ind_dict, price_arr, best_params_global, fine_ranges, n_iterations=50)
        
        if fine_result['sharpe'] > best_sharpe_global:
            best_sharpe_global = fine_result['sharpe']
            best_params_global = fine_result['params']
            all_results.append(fine_result)
            print(f"\n✅ Улучшено после тонкой настройки:")
            print(f"   Sharpe: {best_sharpe_global:.2f}")
            print(f"   Параметры: {best_params_global}")

    # Этап 4: Проверка окрестностей лучших параметров с помощью grid search
    if best_params_global:
        print("\n📊 ЭТАП 4: Локальный grid search вокруг лучших параметров")
        
        # Создаем узкую сетку вокруг лучших параметров
        local_grid = {}
        for key in best_params_global.keys():
            center = best_params_global[key]
            range_size = param_ranges[key][1] - param_ranges[key][0]
            step = range_size * 0.15  # 15% шага
            
            values = [
                max(param_ranges[key][0], round(center - step, 2)),
                max(param_ranges[key][0], round(center, 2)),
                min(param_ranges[key][1], round(center + step, 2))
            ]
            # Удаляем дубликаты
            values = sorted(list(set(values)))
            if len(values) < 2:
                values = [center]
            local_grid[key] = values
        
        print(f"   Локальная сетка: {local_grid}")
        
        grid_best_params, grid_best_sharpe, grid_results, total_cfgs = grid_search_optimization(
            data_dict, ind_dict, price_arr, local_grid, start_cfg=1, debug_filters=True
        )
        
        if grid_best_sharpe > best_sharpe_global:
            best_sharpe_global = grid_best_sharpe
            best_params_global = grid_best_params
            all_results.extend(grid_results)
            print(f"\n✅ Улучшено после локального grid search:")
            print(f"   Sharpe: {best_sharpe_global:.2f}")
            print(f"   Параметры: {best_params_global}")
        else:
            all_results.extend(grid_results)

    # Считаем базовую сетку для сравнения (опционально)
    print("\n📊 ЭТАП 5: Сравнение с базовой сеткой параметров")
    base_grid_result, base_sharpe, base_results, _ = grid_search_optimization(
        data_dict, ind_dict, price_arr, param_grid_base, start_cfg=1, debug_filters=False
    )
    
    if base_results:
        all_results.extend(base_results)
        if base_sharpe > best_sharpe_global:
            best_sharpe_global = base_sharpe
            best_params_global = base_grid_result
            print(f"\n⚠️ Базовая сетка дала лучший результат: Sharpe={base_sharpe:.2f}")

    # ──────────────────────────────────────────────
    # 📊 ИТОГИ
    # ──────────────────────────────────────────────
    if all_results:
        # Преобразуем результаты в DataFrame
        df_res_list = []
        for res in all_results:
            if isinstance(res, dict) and 'Sharpe' in res:
                df_res_list.append(res)
            elif isinstance(res, dict) and 'sharpe' in res:
                df_res_list.append({
                    'Sharpe': round(res['sharpe'], 2),
                    'Return': round(res['return'] * 100, 2),
                    'DD': round(res['dd'] * 100, 2),
                    'Trades': res['trades'],
                    'WinRate': round(res['win_rate'], 1),
                    'Params': res['params']
                })
        
        if df_res_list:
            df_res = pd.DataFrame(df_res_list).sort_values('Sharpe', ascending=False).reset_index(drop=True)
            
            print("\n" + "=" * 60)
            print("🏆 ТОП-10 РЕЗУЛЬТАТОВ АВТО-ОПТИМИЗАЦИИ")
            print("=" * 60)
            print(df_res.head(10).to_string(index=False))
            
            if not df_res.empty:
                best = df_res.iloc[0]
                print(f"\n📌 ЛУЧШИЕ ПАРАМЕТРЫ:")
                for k, v in best['Params'].items():
                    print(f"   {k}: {v}")
                print(f"\n📈 Метрики:")
                print(f"   Sharpe Ratio: {best['Sharpe']:.2f}")
                print(f"   Total Return: {best['Return']:.2f}%")
                print(f"   Max Drawdown: {best['DD']:.2f}%")
                print(f"   Trades: {best['Trades']}")
                print(f"   Win Rate: {best['WinRate']:.1f}%")
            
            # Сохраняем все результаты
            df_res.drop(columns='Params').to_csv('sngs_auto_optimization_full.csv', index=False, encoding='utf-8-sig')
            print("\n💾 Полный отчет сохранен: sngs_auto_optimization_full.csv")
            
            # Сохраняем только топ-20
            df_res.head(20).drop(columns='Params').to_csv('sngs_auto_optimization_top20.csv', index=False, encoding='utf-8-sig')
            print("💾 Топ-20 сохранен: sngs_auto_optimization_top20.csv")
    else:
        print("\n⚠️ Стратегия не совершила ни одной успешной сделки или не прошла фильтры.")
        print("Возможные причины:")
        print("1. Условия входа слишком строгие.")
        print("2. Тренд 1D (EMA10 > EMA32) отсутствовал в выбранные даты.")
        print("3. Диапазоны параметров требуют расширения.")


if __name__ == '__main__':
    # Для Windows обязательна эта конструкция при использовании multiprocessing
    multiprocessing.freeze_support()
    run_optimization()
