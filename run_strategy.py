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
CACHE_FILE = 'sngs_cache.csv'
INIT_CASH = 1_000_000
FEES = 0.001
MIN_TRADES = 1
results = []


def load_data():
    if os.path.exists(CACHE_FILE):
        print("📂 Загружаем данные из кэша...")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = None
    else:
        print("📥 Загружаем SNGS с MOEX (исторические данные)...")
        ticker = moexalgo.Ticker('SNGS')
        start_date = pd.Timestamp('2024-06-01')
        end_date = pd.Timestamp('2024-12-28')
        chunks = []
        current = start_date

        while current < end_date:
            next_m = min(current + pd.DateOffset(months=1), end_date)
            try:
                chunk = ticker.candles(
                    start=current.strftime('%Y-%m-%d'),
                    end=next_m.strftime('%Y-%m-%d'),
                    period='5min'
                )
                if chunk is not None and not chunk.empty:
                    chunks.append(chunk)
                    print(f"  ✓ Загружено: {current.date()} - {next_m.date()} ({len(chunk)} баров)")
            except Exception as e:
                print(f"⚠️ Пропуск чанка {current.date()}: {e}")
            current = next_m

        if not chunks:
            raise ValueError("❌ Данные не получены.")

        raw = pd.concat(chunks, ignore_index=True).sort_index().drop_duplicates()
        idx = 'date' if 'date' in raw.columns else 'begin'
        df = raw.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume', 'value': 'Turnover'
        }).set_index(idx).sort_index()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = None
        df.to_csv(CACHE_FILE)
        print(f"💾 Кэш сохранён: {CACHE_FILE} ({len(df)} баров)")

    needed = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    df = df[[c for c in needed if c in df.columns]].dropna()

    for col in ['Close', 'High', 'Low']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(lower=0.01)

    return df


df = load_data()
price = df['Close'].values.astype(np.float64)

if DEBUG:
    print(f"\n✅ Данные: {len(df)} баров | {df.index[0]} → {df.index[-1]}")
    print(f"📊 Цена: {price.min():.2f} - {price.max():.2f}")


def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def calc_indicators(df, tf_minutes=5):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    ema21 = ema(c, 21)
    ema50 = ema(c, 50)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    macd_line = ema(c, 12) - ema(c, 26)
    macd_hist = macd_line - ema(macd_line, 9)

    k_low, k_high = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = (100 * (c - k_low) / (k_high - k_low).replace(0, np.nan)).rolling(3).mean()

    tr = pd.concat([h - l, h - c.shift(), c.shift() - l], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_lower = bb_mid - 2 * bb_std
    bb_upper = bb_mid + 2 * bb_std

    daily_idx = df.index.floor('D')
    cum_val = (c * v).groupby(daily_idx).cumsum()
    cum_vol = v.groupby(daily_idx).cumsum()
    vwap = (cum_val / cum_vol.replace(0, np.nan)).ffill()

    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    mfv = mfm * v
    cmf = mfv.rolling(20).sum() / v.rolling(20).sum()
    vol_ratio = v / v.rolling(20).mean()

    return pd.DataFrame({
        'ema21': ema21, 'ema50': ema50, 'rsi': rsi, 'macd_hist': macd_hist,
        'stoch_k': stoch_k, 'atr': atr, 'bb_lower': bb_lower, 'bb_upper': bb_upper,
        'vwap': vwap, 'cmf': cmf, 'vol_ratio': vol_ratio
    }).ffill().bfill()


def get_htf_trend(df, tf_mult=12):
    df_htf = df.resample(f'{tf_mult}min').agg({'Close': 'last'}).dropna()
    c = df_htf['Close']
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    di_diff = ema(c, 14) - ema(c, 28)

    htf = pd.Series(0, index=df_htf.index)
    htf[(ema50 > ema200) & (di_diff > 0)] = 1
    htf[(ema50 < ema200) & (di_diff < 0)] = -1
    return htf.reindex(df.index, method='ffill').fillna(0)


def generate_orders(df, ind, htf_trend, params):
    """Генерирует ордера для vectorbt 1.0.0
    
    direction: 0=nothing, 1=long entry/add, 2=short entry/add
    Для закрытия используем size=np.inf с правильным направлением
    """
    c = df['Close']
    p = params
    n = len(c)

    trend_bull = (htf_trend == 1) & (c > ind['ema50'])
    trend_bear = (htf_trend == -1) & (c < ind['ema50'])

    mom_long = (ind['rsi'] < 40) & (ind['macd_hist'] > ind['macd_hist'].shift(1)) & (ind['stoch_k'] < 30)
    mom_short = (ind['rsi'] > 60) & (ind['macd_hist'] < ind['macd_hist'].shift(1)) & (ind['stoch_k'] > 70)

    vol_long = c <= ind['bb_lower']
    vol_short = c >= ind['bb_upper']

    flow_long = (ind['cmf'] > -0.15) & (ind['vol_ratio'] > p['vol_ratio_thresh'])
    flow_short = (ind['cmf'] < 0.15) & (ind['vol_ratio'] > p['vol_ratio_thresh'])

    score_long = trend_bull.astype(int) + mom_long.astype(int) + vol_long.astype(int) + flow_long.astype(int)
    score_short = trend_bear.astype(int) + mom_short.astype(int) + vol_short.astype(int) + flow_short.astype(int)

    entry_long = score_long >= p['confluence_thresh']
    entry_short = score_short >= p['confluence_thresh']

    # Векторные сигналы для vbt
    directions = np.zeros(n, dtype=np.int32)
    sizes = np.full(n, np.nan, dtype=np.float64)
    
    # Отдельные массивы для entries и exits
    long_entries = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    long_exits = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)

    pos = 0
    avg_done = False
    entry_price = np.nan
    sl_price = np.nan
    tp_price = np.nan
    cooldown = 0

    for i in range(n):
        if cooldown > 0:
            cooldown -= 1
            continue

        px = c.iloc[i]
        atr_i = ind['atr'].iloc[i]

        if pos == 0:
            if entry_long.iloc[i]:
                pos = 1
                entry_price = px
                sl_price = px - p['sl_atr_mult'] * atr_i
                tp_price = px + p['tp_atr_mult'] * atr_i
                avg_done = False
                long_entries[i] = True
                sizes[i] = 1.0
            elif entry_short.iloc[i]:
                pos = -1
                entry_price = px
                sl_price = px + p['sl_atr_mult'] * atr_i
                tp_price = px - p['tp_atr_mult'] * atr_i
                avg_done = False
                short_entries[i] = True
                sizes[i] = 1.0

        elif pos == 1:
            if not avg_done and px <= entry_price - p['avg_atr_mult'] * atr_i:
                entry_price = (entry_price + px * p['avg_size']) / (1 + p['avg_size'])
                sl_price = entry_price - p['sl_atr_mult'] * atr_i
                tp_price = entry_price + p['tp_atr_mult'] * atr_i
                avg_done = True
                long_entries[i] = True
                sizes[i] = p['avg_size']
            elif px <= sl_price or px >= tp_price or score_long.iloc[i] < 2:
                long_exits[i] = True
                sizes[i] = 1.0
                pos = 0
                cooldown = 1

        elif pos == -1:
            if not avg_done and px >= entry_price + p['avg_atr_mult'] * atr_i:
                entry_price = (entry_price + px * p['avg_size']) / (1 + p['avg_size'])
                sl_price = entry_price + p['sl_atr_mult'] * atr_i
                tp_price = entry_price - p['tp_atr_mult'] * atr_i
                avg_done = True
                short_entries[i] = True
                sizes[i] = p['avg_size']
            elif px >= sl_price or px <= tp_price or score_short.iloc[i] < 2:
                short_exits[i] = True
                sizes[i] = 1.0
                pos = 0
                cooldown = 1

    # Конвертируем в формат vbt: entries используют direction 1/2, exits используют size=np.inf
    for i in range(n):
        if long_entries[i]:
            directions[i] = 1
        elif short_entries[i]:
            directions[i] = 2
        elif long_exits[i]:
            directions[i] = 2  # Закрываем лонг шортом
            sizes[i] = np.inf  # Закрыть всю позицию
        elif short_exits[i]:
            directions[i] = 1  # Закрываем шорт лонгом
            sizes[i] = np.inf  # Закрыть всю позицию

    return directions, sizes


print("\n⚙️ Расчёт индикаторов и тренда старшего ТФ...")
ind = calc_indicators(df, tf_minutes=5)
htf_trend = get_htf_trend(df, tf_mult=12)

param_grid = {
    'confluence_thresh': [2, 3],
    'vol_ratio_thresh': [1.0, 1.2],
    'sl_atr_mult': [1.0, 1.5],
    'tp_atr_mult': [2.0, 2.5],
    'avg_atr_mult': [1.0, 1.5],
    'avg_size': [0.5]
}

best_sharpe = -999
best_params = None
error_count = 0
cfg_count = 0

print("🔄 Запуск сеточной оптимизации...\n")
for vals in product(*param_grid.values()):
    p = dict(zip(param_grid.keys(), vals))
    cfg_count += 1

    try:
        directions, sizes = generate_orders(df, ind, htf_trend, p)

        active_mask = ~np.isnan(sizes)
        if np.any(active_mask):
            assert np.all(np.isin(directions[active_mask], [0, 1, 2])), "direction содержит недопустимые значения"
            assert np.all(sizes[active_mask] > 0), "Активные размеры ордеров должны быть > 0"

        dirs_arr = np.ascontiguousarray(directions)
        sizes_arr = np.ascontiguousarray(sizes)
        close_arr = np.ascontiguousarray(price)

        pf = vbt.Portfolio.from_orders(
            close=close_arr,
            direction=dirs_arr,
            size=sizes_arr,
            size_type='percent',
            init_cash=float(INIT_CASH),
            fees=FEES,
            log=False,
            allow_partial=True,
            cash_sharing=False,
            lock_cash=False
        )

        t = int(pf.trades.count())
        sr = float(pf.sharpe_ratio)
        dd = float(pf.max_drawdown)
        ret = float(pf.total_return)
        wr = float(pf.trades.win_rate()) * 100 if t > 0 else 0

        if t < MIN_TRADES:
            continue

        if pd.isna(sr) or pd.isinf(sr) or dd > 0.30:
            continue

        if sr > best_sharpe:
            best_sharpe = sr
            best_params = p
            results.append({
                'Стратегия': 'Multi_Confluence_v5',
                'Доходность %': round(ret * 100, 2),
                'Sharpe': round(sr, 2),
                'Макс. просадка %': round(dd * 100, 2),
                'Винрейт %': round(wr, 2),
                'Сделок': t,
                'Params': p
            })
            print(f"✓ Конфиг #{cfg_count}: SR={sr:.2f} | DD={dd*100:.1f}% | Сделок={t} | Ret={ret*100:.1f}%")

    except Exception as e:
        error_count += 1
        print(f"\n❌ Ошибка в конфиге #{cfg_count} {p}:")
        traceback.print_exc()
        break

if results:
    df_res = pd.DataFrame(results).sort_values('Sharpe', ascending=False).reset_index(drop=True)
    print(f"\n{'='*60}")
    print(f"✅ Оптимизировано: {len(results)} конфигураций (ошибок: {error_count})")
    print("🏆 Топ-5 по Sharpe Ratio:")
    print(df_res.head(5).drop(columns='Params').to_string(index=False))

    best_row = df_res.iloc[0]
    print(f"\n📌 Лучшие параметры: {best_row['Params']}")
    print(f"📈 Sharpe: {best_row['Sharpe']} | DD: {best_row['Макс. просадка %']}% | Сделок: {best_row['Сделок']}")

    df_res.drop(columns='Params').to_csv('sngs_confluence_optimization.csv', index=False, encoding='utf-8-sig')
    print("\n💾 Отчёт сохранён: sngs_confluence_optimization.csv")
else:
    print(f"\n⚠️ Не найдено конфигураций. (Всего ошибок: {error_count})")
    if error_count == 0:
        print("💡 Стратегия работает, но не прошла фильтры.")
