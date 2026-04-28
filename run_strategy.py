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


def get_htf_trend(df):
    """Рассчитывает тренд на дневном таймфрейме (1D)
    
    Простая логика: цена выше SMA20 = бычий, ниже = медвежий
    """
    # Ресемплинг в 1 день
    df_htf = df.resample('1D').agg({'Close': 'last'}).dropna()
    c = df_htf['Close']
    
    # Простая скользящая средняя
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()

    htf = pd.Series(0, index=df_htf.index)
    # Бычий тренд: цена выше SMA20 И SMA20 > SMA50
    htf[(c > sma20) & (sma20 > sma50)] = 1
    # Медвежий тренд: цена ниже SMA20 И SMA20 < SMA50
    htf[(c < sma20) & (sma20 < sma50)] = -1
    
    # Возвращаем к исходному индексу (5 мин), заполняя пропуски последним известным значением
    return htf.reindex(df.index, method='ffill').fillna(0)


def generate_orders(df, ind, htf_trend, params):
    """Генерирует ордера для vectorbt 1.0.0
    
    direction: 2 = Both (автоматически определяет покупку/продажу по позиции)
    size: NaN = нет ордера, >0 = размер ордера в % от капитала
    
    Стратегия: Mean Reversion - покупка на просадках, продажа на росте
    Работает независимо от тренда (торговля в диапазоне)
    
    ВАЖНО: Проверка SL/TP происходит по High/Low бара, а не по Close
    """
    c = df['Close']
    h = df['High']
    l = df['Low']
    p = params
    n = len(c)

    # ТОРГОВЛЯ В ДИАПАЗОНЕ - игнорируем тренд
    # Покупка у нижней границы Bollinger ИЛИ при низком RSI
    entry_long = (c <= ind['bb_lower']) | (ind['rsi'] < 35)
    
    # Продажа у верхней границы Bollinger ИЛИ при высоком RSI
    entry_short = (c >= ind['bb_upper']) | (ind['rsi'] > 65)

    # Векторные сигналы для vbt
    long_entries = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    long_exits = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)
    
    sizes = np.full(n, np.nan, dtype=np.float64)

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

        px_close = c.iloc[i]
        px_low = l.iloc[i]
        px_high = h.iloc[i]
        atr_i = ind['atr'].iloc[i]
        
        # Защита от нулевого ATR
        if pd.isna(atr_i) or atr_i <= 0:
            atr_i = (ind['atr'].iloc[max(0, i-10):i+1].median())
            if pd.isna(atr_i) or atr_i <= 0:
                atr_i = px_close * 0.02

        if pos == 0:
            # ВХОД В ПОЗИЦИЮ (по цене закрытия)
            if entry_long.iloc[i]:
                pos = 1
                entry_price = px_close
                sl_price = px_close - p['sl_atr_mult'] * atr_i
                tp_price = px_close + p['tp_atr_mult'] * atr_i
                avg_done = False
                long_entries[i] = True
                sizes[i] = 1.0
            elif entry_short.iloc[i]:
                pos = -1
                entry_price = px_close
                sl_price = px_close + p['sl_atr_mult'] * atr_i
                tp_price = px_close - p['tp_atr_mult'] * atr_i
                avg_done = False
                short_entries[i] = True
                sizes[i] = 1.0

        elif pos == 1:
            # ЛОНГ позиция
            # ПРОВЕРКА SL/TP ПО LOW/HIGH БАРА!
            sl_hit = px_low <= sl_price
            tp_hit = px_high >= tp_price
            
            # Усреднение (только если еще не было и SL/TP не задеты)
            if not avg_done and not sl_hit and not tp_hit and px_close <= entry_price - p['avg_atr_mult'] * atr_i:
                entry_price = (entry_price + px_close * p['avg_size']) / (1 + p['avg_size'])
                sl_price = entry_price - p['sl_atr_mult'] * atr_i
                tp_price = entry_price + p['tp_atr_mult'] * atr_i
                avg_done = True
                long_entries[i] = True
                sizes[i] = p['avg_size']
            # ВЫХОД из лонга: SL или TP (приоритет SL если оба задеты)
            elif sl_hit or tp_hit:
                long_exits[i] = True
                sizes[i] = 1.0
                pos = 0
                cooldown = 0

        elif pos == -1:
            # ШОРТ позиция
            # ПРОВЕРКА SL/TP ПО HIGH/LOW БАРА!
            sl_hit = px_high >= sl_price
            tp_hit = px_low <= tp_price
            
            # Усреднение (только если еще не было и SL/TP не задеты)
            if not avg_done and not sl_hit and not tp_hit and px_close >= entry_price + p['avg_atr_mult'] * atr_i:
                entry_price = (entry_price + px_close * p['avg_size']) / (1 + p['avg_size'])
                sl_price = entry_price + p['sl_atr_mult'] * atr_i
                tp_price = entry_price - p['tp_atr_mult'] * atr_i
                avg_done = True
                short_entries[i] = True
                sizes[i] = p['avg_size']
            # ВЫХОД из шорта: SL или TP (приоритет SL если оба задеты)
            elif sl_hit or tp_hit:
                short_exits[i] = True
                sizes[i] = 1.0
                pos = 0
                cooldown = 0

    # Конвертируем в формат vbt 1.0.0
    final_directions = np.full(n, 2, dtype=np.int32)
    final_sizes = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if long_entries[i]:
            final_directions[i] = 2
            final_sizes[i] = sizes[i]
        elif short_entries[i]:
            final_directions[i] = 2
            final_sizes[i] = sizes[i]
        elif long_exits[i]:
            final_directions[i] = 2
            final_sizes[i] = 1.0
        elif short_exits[i]:
            final_directions[i] = 2
            final_sizes[i] = 1.0

    return final_directions, final_sizes


print("\n⚙️ Расчёт индикаторов и тренда старшего ТФ (1D)...")
ind = calc_indicators(df, tf_minutes=5)
htf_trend = get_htf_trend(df)  # Теперь используется дневной тренд

# Отладка: проверка распределения тренда
print(f"Тренд: {(htf_trend == 1).sum()} бычьих, {(htf_trend == -1).sum()} медвежьих, {(htf_trend == 0).sum()} нейтральных баров")

# Агрессивная сетка параметров для частой торговли
param_grid = {
    'confluence_thresh': [1],              # Минимальный порог (фактически не используется в новой логике)
    'vol_ratio_thresh': [1.0],             # Не используется в новой логике
    'sl_atr_mult': [1.5, 2.0, 2.5],        # Стоп-лосс в ATR
    'tp_atr_mult': [2.0, 3.0, 4.0],        # Тейк-профит в ATR
    'avg_atr_mult': [1.5, 2.0],            # Усреднение по ATR
    'avg_size': [0.5]                      # Размер усреднения
}

total_configs = len(list(product(*param_grid.values())))
print(f"📊 Всего конфигураций для проверки: {total_configs}\n")

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
            assert np.all(np.isin(directions[active_mask], [0, 1, 2])), "direction содержит недопустимые значения (только 0, 1 или 2)"
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
            lock_cash=False,
            freq='5min'  # Указываем частоту данных для правильного расчета метрик
        )

        t = pf.trades.count()
        sr = pf.sharpe_ratio()
        dd = pf.max_drawdown()
        ret = pf.total_return()
        wr = pf.trades.win_rate() * 100 if t > 0 else 0

        if t < MIN_TRADES:
            continue

        if np.isnan(sr) or np.isinf(sr) or dd > 0.30:
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
