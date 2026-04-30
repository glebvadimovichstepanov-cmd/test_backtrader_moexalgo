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
MIN_TRADES = 5
results = []


# ──────────────────────────────────────────────
# 📥 ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────
def load_data():
    if os.path.exists(CACHE_FILE):
        print("📂 Загружаем данные из кэша...")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = None
    else:
        print("📥 Загружаем SNGS с MOEX...")
        ticker = moexalgo.Ticker('SNGS')
        # Расширим диапазон для лучшего теста
        start_date = pd.Timestamp('2025-01-01')
        end_date = pd.Timestamp('2026-04-28')
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

    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in needed if c in df.columns]].dropna()

    # Защита от NaN/Inf
    for col in ['Close', 'High', 'Low']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(lower=0.01)

    return df


df = load_data()
price = df['Close'].values.astype(np.float64)

if DEBUG:
    print(f"✅ Данные: {len(df)} баров | {df.index[0]} → {df.index[-1]}")
    print(f"📊 Цена: {price.min():.2f} - {price.max():.2f}")


# ──────────────────────────────────────────────
# 🧮 1. ИНДИКАТОРЫ
# ──────────────────────────────────────────────
def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def calc_indicators(df, tf_minutes=5):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    ema21 = ema(c, 21)
    ema50 = ema(c, 50)

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    # MACD
    macd_line = ema(c, 12) - ema(c, 26)
    macd_hist = macd_line - ema(macd_line, 9)

    # Stochastic
    k_low, k_high = l.rolling(14).min(), h.rolling(14).max()
    stoch_k = (100 * (c - k_low) / (k_high - k_low).replace(0, np.nan)).rolling(3).mean()

    # ATR & Bollinger
    tr = pd.concat([h - l, h - c.shift(), c.shift() - l], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_lower = bb_mid - 2 * bb_std
    bb_upper = bb_mid + 2 * bb_std

    # Volume Flow
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


def get_htf_trend(df, short_win=10, long_win=32):
    """
    Расчет тренда на старшем ТФ (1D).
    Окна уменьшены до 10 и 32 для быстрой реакции.
    """
    # Ресемплинг в 1 день
    df_htf = df.resample('1D').agg({'Close': 'last'}).dropna()
    c = df_htf['Close']

    ema_short = ema(c, short_win)
    ema_long = ema(c, long_win)

    # Бычий тренд: Цена > EMA10 > EMA32
    bull = (c > ema_short) & (ema_short > ema_long)
    # Медвежий тренд: Цена < EMA10 < EMA32 (используем для фильтра, хотя шортов нет)
    bear = (c < ema_short) & (ema_short < ema_long)

    htf = pd.Series(0, index=df_htf.index)
    htf[bull] = 1
    htf[bear] = -1

    # Возвращаем к исходному таймфрейму (5мин)
    return htf.reindex(df.index, method='ffill').fillna(0)


# ──────────────────────────────────────────────
# 📤 2. ГЕНЕРАЦИЯ ОРДЕРОВ (ONLY LONG)
# ──────────────────────────────────────────────
def generate_orders(df, ind, htf_trend, params):
    c = df['Close']
    h = df['High']
    l = df['Low']
    p = params
    n = len(c)

    # 1. ФИЛЬТР ТРЕНДА (HTF)
    # Торгуем ЛОНГ только если HTF тренд = 1
    trend_ok = (htf_trend == 1)

    # 2. СИГНАЛЫ ВХОДА (LONG ONLY)
    # Моментум: RSI перепроданность + разворот MACD
    mom_long = (ind['rsi'] < 40) & (ind['macd_hist'] > ind['macd_hist'].shift(1))

    # Волатильность: Пробой нижней границы BB
    vol_long = c <= ind['bb_lower']

    # Поток: CMF не слишком негативный + Объем выше среднего
    flow_long = (ind['cmf'] > -0.15) & (ind['vol_ratio'] > p.get('vol_ratio_thresh', 1.0))

    # Счет конфлюэнции
    score_long = mom_long.astype(int) + vol_long.astype(int) + flow_long.astype(int)
    entry_signal = trend_ok & (score_long >= p['confluence_thresh'])

    # 3. ФИЛЬТР ВОЛАТИЛЬНОСТИ (ATR)
    atr_med = ind['atr'].median()
    vol_filter = (ind['atr'] > 0.5 * atr_med) & (ind['atr'] < 3.0 * atr_med)

    # Итоговый сигнал входа
    final_entry = entry_signal & vol_filter

    # ─── МАШИНА СОСТОЯНИЙ (ONLY LONG) ───
    # direction: 1 = Buy, 0 = Hold/Close (для выхода используем size=0 или отдельный сигнал sell)
    # В vbt Portfolio.from_orders:
    # direction=1, size=X -> Покупка
    # direction=-1, size=X -> Продажа (Шорт или закрытие лонга)
    # direction=0 -> Ничего

    directions = np.zeros(n, dtype=np.int32)  # По умолчанию 0 (ничего не делаем)
    sizes = np.full(n, np.nan, dtype=np.float64)

    pos = 0  # 0 = нет позиции, 1 = в лонге
    avg_done = False  # Было ли усреднение в этой сделке
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    cooldown = 0

    for i in range(n):
        if cooldown > 0:
            cooldown -= 1
            continue

        px_close = c.iloc[i]
        px_high = h.iloc[i]
        px_low = l.iloc[i]
        atr_i = ind['atr'].iloc[i]

        # ПРОВЕРКА ВЫХОДА ИЗ ПОЗИЦИИ (при наличии позиции)
        if pos == 1:
            # Проверка срабатывания SL/TP по экстремумам бара
            hit_sl = px_low <= sl_price
            hit_tp = px_high >= tp_price

            # Сигнал на выход по ухудшению условий (например, счет упал ниже 1)
            exit_signal = score_long.iloc[i] < 1

            if hit_sl or hit_tp or exit_signal:
                # ЗАКРЫВАЕМ ПОЗИЦИЮ
                directions[i] = -1  # Sell
                sizes[i] = 1.0  # Закрываем весь объем (100% от начального)
                pos = 0
                avg_done = False
                cooldown = 2  # Небольшая пауза после выхода
                continue

            # ПРОВЕРКА УСРЕДНЕНИЯ (если еще не было и цена упала)
            if not avg_done and px_close <= entry_price - p['avg_atr_mult'] * atr_i:
                # Докупаем (усредняемся)
                # Новая средняя цена
                old_qty = 1.0
                new_qty = p['avg_size']
                entry_price = (entry_price * old_qty + px_close * new_qty) / (old_qty + new_qty)

                # Пересчет SL/TP от новой средней
                sl_price = entry_price - p['sl_atr_mult'] * atr_i
                tp_price = entry_price + p['tp_atr_mult'] * atr_i

                avg_done = True
                directions[i] = 1  # Buy
                sizes[i] = p['avg_size']  # Докупаем часть
                continue

        # ПРОВЕРКА ВХОДА (если нет позиции)
        if pos == 0:
            if final_entry.iloc[i]:
                # ОТКРЫВАЕМ ПОЗИЦИЮ
                pos = 1
                entry_price = px_close
                sl_price = entry_price - p['sl_atr_mult'] * atr_i
                tp_price = entry_price + p['tp_atr_mult'] * atr_i
                avg_done = False

                directions[i] = 1  # Buy
                sizes[i] = 1.0  # 100% размера ордера
                # Cooldown не ставим, чтобы можно было сразу усредниться, если цена пойдет против

    return directions, sizes


# ──────────────────────────────────────────────
# 🔍 3. ОПТИМИЗАЦИЯ
# ──────────────────────────────────────────────
print("⚙️ Расчёт индикаторов и тренда (1D, окна 10/32)...")
ind = calc_indicators(df, tf_minutes=5)
htf_trend = get_htf_trend(df, short_win=10, long_win=32)

# Убрали параметры, связанные с шортами
param_grid = {
    'confluence_thresh': [2, 3],  # Сколько условий должно совпасть
    'vol_ratio_thresh': [1.0, 1.5],  # Порог объема
    'sl_atr_mult': [1.5, 2.0, 2.5],  # Стоп-лосс
    'tp_atr_mult': [2.0, 3.0, 4.0],  # Тейк-профит
    'avg_atr_mult': [1.5, 2.0],  # Дистанция усреднения
    'avg_size': [0.5]  # Размер усреднения (50% от лота)
}

best_sharpe = -999
best_params = None
error_count = 0
cfg_count = 0

print("🔄 Запуск сеточной оптимизации (Long Only)...")
for vals in product(*param_grid.values()):
    p = dict(zip(param_grid.keys(), vals))
    cfg_count += 1

    try:
        directions, sizes = generate_orders(df, ind, htf_trend, p)

        # Валидация
        active_mask = ~np.isnan(sizes)
        if active_mask.sum() == 0:
            continue  # Нет сделок

        # Подготовка массивов
        dirs_arr = np.ascontiguousarray(directions)
        sizes_arr = np.ascontiguousarray(sizes)
        close_arr = np.ascontiguousarray(price)

        pf = vbt.Portfolio.from_orders(
            close=close_arr,
            direction=dirs_arr,
            size=sizes_arr,
            size_type='percent',  # Проценты от доступного кэша/акций
            init_cash=float(INIT_CASH),
            fees=FEES,
            log=False,
            allow_partial=True,
            cash_sharing=False,
            lock_cash=False
        )

        t = int(pf.trades.count)
        sr = float(pf.sharpe_ratio)
        dd = float(pf.max_drawdown)
        ret = float(pf.total_return)
        wr = float(pf.trades.win_rate) * 100 if t > 0 else 0

        # Фильтры качества
        if t < MIN_TRADES:
            continue
        if pd.isna(sr) or pd.isinf(sr) or dd > 0.40:  # Макс просадка 40%
            continue

        if sr > best_sharpe:
            best_sharpe = sr
            best_params = p
            results.append({
                'Стратегия': 'LongOnly_1D_Fast',
                'Доходность %': round(ret * 100, 2),
                'Sharpe': round(sr, 2),
                'Макс. просадка %': round(dd * 100, 2),
                'Винрейт %': round(wr, 2),
                'Сделок': t,
                'Params': p
            })
            if DEBUG:
                print(f"  🏆 Конфиг #{cfg_count}: Sharpe={sr:.2f} | Ret={ret:.2%} | Params={p}")

    except Exception as e:
        error_count += 1
        if DEBUG:
            print(f"\n❌ Ошибка в конфиге #{cfg_count}: {e}")
        # traceback.print_exc()

# ──────────────────────────────────────────────
# 📊 4. ИТОГОВЫЙ ВЫВОД
# ──────────────────────────────────────────────
if results:
    df_res = pd.DataFrame(results).sort_values('Sharpe', ascending=False).reset_index(drop=True)
    print(f"\n✅ Оптимизировано: {len(results)} конфигураций (ошибок: {error_count})")
    print("🏆 Топ-3 по Sharpe Ratio:")
    print(df_res.head(3).drop(columns='Params').to_string(index=False))

    best_row = df_res.iloc[0]
    print(f"\n📌 Лучшие параметры: {best_row['Params']}")
    print(f"📈 Sharpe: {best_row['Sharpe']} | DD: {best_row['Макс. просадка %']}% | Сделок: {best_row['Сделок']}")

    df_res.drop(columns='Params').to_csv('sngs_long_only_opt.csv', index=False, encoding='utf-8-sig')
    print("💾 Отчёт сохранён: sngs_long_only_opt.csv")
else:
    print(f"\n⚠️ Не найдено подходящих конфигураций. (Ошибок: {error_count})")
    print("💡 Попробуйте снизить MIN_TRADES или расширить диапазоны параметров.")