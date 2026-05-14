#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прогнозирование + генерация торгового сигнала для интрадей-робота.

Улучшения для надёжного сигнала:
  1. Мультитаймфреймовый консенсус с весами (10T > 1H > 1D для интрадей)
  2. Фильтр качества сигнала: требуем минимальный R/R и узкий интервал
  3. Фильтр торговой сессии MOEX (не торгуем в первые/последние 30 мин)
  4. Паттерн-фильтр: сигнал только при согласии модели и индикаторов
  5. Confidence score 0-100 с порогом входа
  6. Стоп-лосс и тейк-профит из квантилей прогноза
  7. Ensemble: запускаем модель N раз и усредняем (снижает шум)
"""

import os
import sys
import logging
import warnings
import traceback
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import torch
import requests

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.evaluation.backtest import make_evaluation_predictions
    logging.getLogger("gluonts").setLevel(logging.ERROR)
except ImportError:
    print("Lag-Llama не установлен.")
    sys.exit(1)

import moexalgo

# ── PyTorch 2.6: weights_only patch ───────────────────────────────────────
_orig_load = torch.load
def _patched_load(f, map_location=None, pickle_module=None,
                  weights_only=False, mmap=None, **kw):
    return _orig_load(f, map_location=map_location, pickle_module=pickle_module,
                      weights_only=False, **({} if mmap is None else {"mmap": mmap}), **kw)
torch.load = _patched_load
# ──────────────────────────────────────────────────────────────────────────

pd.set_option("mode.chained_assignment", None)
np.random.seed(42)
torch.manual_seed(42)

# ======================== НАСТРОЙКИ ========================

#TICKER        = "SNGS"
CACHE_DIR     = "cache_data"
RESULTS_DIR   = "results"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

START_DATE = pd.Timestamp("2022-01-01")
END_DATE   = pd.Timestamp("2026-06-01")

PREDICTION_STEPS = 10
CONTEXT_LENGTH   = 1092
NUM_SAMPLES      = 200   # больше семплов = стабильнее квантили
ENSEMBLE_RUNS    = 3     # запускаем модель N раз и усредняем (снижает шум)

# Веса таймфреймов для интрадей (короткие важнее)
TF_WEIGHTS = {"10T": 0.50, "1H": 0.35, "1D": 0.15}
TIMEFRAMES  = {"1D": "1d", "1H": "1h", "10T": "10min"}

# Параметры торгового фильтра
MIN_CONFIDENCE    = 60    # минимальный score 0-100 для входа
MIN_EXPECTED_MOVE = 0.15  # минимальное ожидаемое движение в % (иначе шум)
MAX_INTERVAL_PCT  = 2.0   # максимальная ширина интервала в % от цены
MIN_RR_RATIO      = 1.5   # минимальный Risk/Reward

# Торговая сессия MOEX
SESSION_START = dtime(10, 30)   # не входим в первые 30 мин
SESSION_END   = dtime(18, 30)   # не входим в последние 30 мин

LLM_API_URL = "http://127.0.0.1:8080/completion"
LLM_ENABLED = True

LAG_LLAMA_CHECKPOINT_PATH = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}")

# ======================== ЛОГИРОВАНИЕ ========================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("sngs_signal")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler("signal_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ======================== ЗАГРУЗКА ДАННЫХ ========================

def load_data_tf(ticker_name: str, start: pd.Timestamp, end: pd.Timestamp,
                 logger: logging.Logger) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Загрузка данных с MOEX с использованием кэша и догрузкой недостающих свечей.
    
    Логика:
    1. Проверяем наличие кэш-файла
    2. Если есть — загружаем и проверяем диапазон дат
    3. Если кэш устарел или его нет — догружаем недостающие периоды
    4. Сохраняем обновлённый кэш
    """
    tf_map = {"1D": "1d", "1H": "1h", "10T": "10min"}
    date_ranges = pd.date_range(start=start, end=end, freq="30D")
    ranges = [(date_ranges[i], date_ranges[i+1]) for i in range(len(date_ranges)-1)]
    if date_ranges[-1] < end:
        ranges.append((date_ranges[-1], end))

    dataframes = {}
    for tf_name, tf_code in tf_map.items():
        cache_file = os.path.join(
            CACHE_DIR,
            f"{ticker_name}_{tf_name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        )
        
        # Попытка загрузить из кэша
        df_cached = None
        if os.path.exists(cache_file):
            try:
                logger.info(f"[{tf_name}] Загрузка из кэша...")
                df_cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df_cached.empty:
                    # Проверяем актуальность кэша
                    last_cached_date = df_cached.index.max()
                    now = pd.Timestamp.now()
                    cache_age_hours = (now - last_cached_date).total_seconds() / 3600
                    
                    # Для интрадей таймфреймов — кэш не старше 1 часа
                    max_age = 1 if tf_name in ("10T", "1H") else 24
                    
                    if cache_age_hours < max_age and last_cached_date >= end:
                        logger.info(f"[{tf_name}] Кэш актуален ({len(df_cached)} баров)")
                        dataframes[tf_name] = df_cached
                        continue
                    else:
                        logger.info(f"[{tf_name}] Кэш устарел (последняя запись: {last_cached_date}), догружаем...")
                else:
                    logger.warning(f"[{tf_name}] Кэш пуст, загружаем заново")
            except Exception as e:
                logger.warning(f"[{tf_name}] Ошибка чтения кэша: {e}")
        
        # Загрузка с MOEX
        logger.info(f"[{tf_name}] Загрузка с MOEX...")
        ticker = moexalgo.Ticker(ticker_name)
        chunk_dfs = []
        
        # Если есть кэш, определяем дату начала догрузки
        if df_cached is not None and not df_cached.empty:
            last_cached_date = df_cached.index.max()
            # Начинаем загрузку с даты последнего кэша
            ranges_to_load = [(s, e) for s, e in ranges if s >= last_cached_date]
            if not ranges_to_load and last_cached_date < end:
                # Добавляем последний диапазон если он частично покрыт
                ranges_to_load = [(last_cached_date, end)]
        else:
            ranges_to_load = ranges
        
        if ranges_to_load:
            for chunk_start, chunk_end in ranges_to_load:
                try:
                    df_chunk = ticker.candles(
                        start=chunk_start.strftime("%Y-%m-%d"),
                        end=chunk_end.strftime("%Y-%m-%d"),
                        period=tf_code
                    )
                    if df_chunk is not None and not df_chunk.empty:
                        idx_col = "date" if "date" in df_chunk.columns else "begin"
                        df_chunk = df_chunk.set_index(idx_col)
                        df_chunk = df_chunk.rename(columns={
                            "open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"
                        })
                        df_chunk.index = pd.to_datetime(df_chunk.index)
                        chunk_dfs.append(df_chunk)
                        logger.debug(f"  [{tf_name}] Загружен чанк {chunk_start} - {chunk_end}: {len(df_chunk)} баров")
                except Exception as e:
                    logger.warning(f"  Ошибка чанка [{tf_name}] {chunk_start}: {e}")
        else:
            logger.debug(f"[{tf_name}] Все данные уже в кэше")
        
        # Объединяем кэш и новые данные
        if chunk_dfs:
            df_new = pd.concat(chunk_dfs).sort_index()
            df_new = df_new[~df_new.index.duplicated(keep="first")]
            
            if df_cached is not None and not df_cached.empty:
                # Удаляем дубликаты между кэшем и новыми данными
                df_cached = df_cached[df_cached.index < df_new.index.min()]
                df = pd.concat([df_cached, df_new]).sort_index()
            else:
                df = df_new
            
            # Сохраняем обновлённый кэш
            df.to_csv(cache_file)
            logger.info(f"[{tf_name}] Кэш обновлён: {len(df)} баров")
        elif df_cached is not None and not df_cached.empty:
            df = df_cached
            logger.info(f"[{tf_name}] Используем кэш: {len(df)} баров")
        else:
            logger.error(f"[{tf_name}] Нет данных")
            continue

        needed = ["Open", "High", "Low", "Close", "Volume"]
        df = df[needed].dropna()
        for col in ["Close", "High", "Low"]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(lower=0.01)

        dataframes[tf_name] = df
        logger.info(f"[{tf_name}] Итого: {len(df)} баров")

    return dataframes if dataframes else None

# ======================== ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ========================

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)

def compute_macd(series: pd.Series) -> Tuple[float, float, str]:
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    # Определяем направление и тип пересечения
    cross = "bullish" if float(macd.iloc[-1]) > float(signal.iloc[-1]) else "bearish"
    return round(float(macd.iloc[-1]), 4), round(float(signal.iloc[-1]), 4), cross

def compute_bb(series: pd.Series, period: int = 20) -> Tuple[float, float, float, str]:
    ma    = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = float((ma + 2*std).iloc[-1])
    lower = float((ma - 2*std).iloc[-1])
    price = float(series.iloc[-1])
    pos   = (price - lower) / (upper - lower) if upper != lower else 0.5
    if price > upper:
        zone = "overbought"
    elif price < lower:
        zone = "oversold"
    elif pos < 0.2:
        zone = "near_lower"
    elif pos > 0.8:
        zone = "near_upper"
    else:
        zone = "mid"
    return upper, lower, pos, zone

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range — мера волатильности для стоп-лосса"""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 4)

def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX — сила тренда. >25 = сильный тренд, <20 = флэт"""
    h, l, c = df["High"], df["Low"], df["Close"]
    up   = h - h.shift()
    down = l.shift() - l
    tr   = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    plus_dm  = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr14    = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr14
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr14
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()
    return round(float(adx.iloc[-1]), 1)

def get_volume_trend(df: pd.DataFrame, window: int = 20) -> str:
    recent = df["Volume"].iloc[-window:].mean()
    older  = df["Volume"].iloc[-2*window:-window].mean()
    if recent > older * 1.2:
        return "растущий"
    elif recent < older * 0.8:
        return "падающий"
    return "нейтральный"

def detect_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
    """Простые уровни поддержки/сопротивления через локальные экстремумы"""
    recent = df["Close"].iloc[-lookback:]
    support    = float(recent.min())
    resistance = float(recent.max())
    return support, resistance

# ======================== ПРЕДОБРАБОТКА ========================

def prepare_log_returns(df: pd.DataFrame, target_col: str = "Close") -> pd.Series:
    series = df[target_col].copy()
    lr = np.log(series / series.shift(1)).dropna()
    return lr.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# ======================== ЧЕКПОИНТ ========================

_CKPT_CACHE: Dict = {}

def get_checkpoint_path(model_path: Optional[str] = None) -> str:
    if model_path and os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="time-series-foundation-models/Lag-Llama",
        filename="lag-llama.ckpt", repo_type="model"
    )

def load_checkpoint_data(path: str) -> dict:
    if path not in _CKPT_CACHE:
        _CKPT_CACHE[path] = torch.load(path, map_location="cpu", weights_only=False)
    return _CKPT_CACHE[path]

# ======================== ПРОГНОЗИРОВАНИЕ ========================

def _single_predict(checkpoint_path: str, series_ctx: pd.Series,
                    pd_freq: str, prediction_steps: int,
                    actual_context: int, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Один прогон модели, возвращает (mean, samples)"""
    dataset = PandasDataset({"target": series_ctx.astype("float32")}, freq=pd_freq)

    ckpt_data    = load_checkpoint_data(checkpoint_path)
    hparams      = ckpt_data.get("hyper_parameters", {})
    model_kwargs = hparams.get("model_kwargs", {})

    SKIP = {"distr_output", "rope_scaling", "lags_seq",
            "input_size", "context_length", "prediction_length"}
    ARCH = {"n_embd_per_head", "n_layer", "n_head", "max_context_length",
            "scaling", "time_feat", "num_parallel_samples", "dropout"}

    ekw = dict(ckpt_path=checkpoint_path, prediction_length=prediction_steps,
               context_length=actual_context, num_parallel_samples=num_samples,
               device=DEVICE, batch_size=1)
    for k, v in model_kwargs.items():
        if k in ARCH and k not in SKIP:
            ekw[k] = v

    est  = LagLlamaEstimator(**ekw)
    pred = est.create_predictor(est.create_transformation(), est.create_lightning_module())

    fc_it, _ = make_evaluation_predictions(dataset=dataset, predictor=pred, num_samples=num_samples)
    fcs = list(fc_it)
    if not fcs:
        raise ValueError("Нет прогнозов")

    fc = fcs[0]
    return fc.mean, fc.samples  # samples: (num_samples, prediction_steps)


def predict_ensemble(checkpoint_path: str, series: pd.Series, freq: str,
                     prediction_steps: int, context_length: int,
                     num_samples: int, n_runs: int = 3
                     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float]:
    """
    Запускает модель n_runs раз и усредняет результаты.
    Возвращает: (mean_pred, (q10, q90), uncertainty_score)
    uncertainty_score — среднее стд между запусками (0 = стабильно, >0.01 = нестабильно)
    """
    freq_map = {"1d": "D", "1h": "h", "10min": "10min", "5min": "5min", "1min": "min"}
    pd_freq  = freq_map.get(freq, freq)

    series = series.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    actual_context = min(len(series), context_length)
    series_ctx = series.tail(actual_context).copy()
    series_ctx.index = pd.DatetimeIndex(series_ctx.index)
    try:
        series_ctx = series_ctx.asfreq(pd_freq, method="ffill")
    except Exception:
        pass

    all_means   = []
    all_samples = []

    for run in range(n_runs):
        torch.manual_seed(42 + run)   # разные seed → разные семплы
        mean, samples = _single_predict(
            checkpoint_path, series_ctx, pd_freq,
            prediction_steps, actual_context, num_samples
        )
        all_means.append(mean)
        all_samples.append(samples)

    # Усредняем по всем запускам
    mean_pred = np.mean(all_means, axis=0)
    all_s     = np.concatenate(all_samples, axis=0)  # (n_runs*num_samples, steps)

    q10 = np.quantile(all_s, 0.10, axis=0)
    q90 = np.quantile(all_s, 0.90, axis=0)

    # Нестабильность = среднее стандартное отклонение mean между запусками по шагам
    uncertainty = float(np.std(all_means, axis=0).mean())

    return mean_pred, (q10, q90), uncertainty


# ======================== ДЕНОРМАЛИЗАЦИЯ ========================

def denormalize(preds: np.ndarray, last_price: float) -> np.ndarray:
    return last_price * np.exp(np.cumsum(np.concatenate([[0], preds])))[1:]

# ======================== ТОРГОВЫЙ СИГНАЛ ========================

def compute_confidence_score(
        pred_prices: np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
        last_price: float,
        rsi: float,
        macd_cross: str,
        bb_zone: str,
        adx: float,
        volume_trend: str,
        uncertainty: float,
        tf_weight: float
) -> Tuple[int, Dict]:
    """
    Вычисляет confidence score 0-100 для торгового сигнала.
    Возвращает (score, breakdown) где breakdown — компоненты оценки.
    """
    scores = {}

    # 1. Направление прогноза (30 очков)
    final_move_pct = (pred_prices[-1] / last_price - 1) * 100
    if abs(final_move_pct) >= MIN_EXPECTED_MOVE:
        scores["direction"] = 30
    elif abs(final_move_pct) >= MIN_EXPECTED_MOVE / 2:
        scores["direction"] = 15
    else:
        scores["direction"] = 0

    # 2. Ширина интервала (20 очков) — узкий интервал = уверенность
    interval_pct = (q_high[-1] - q_low[-1]) / last_price * 100
    if interval_pct <= MAX_INTERVAL_PCT * 0.5:
        scores["interval"] = 20
    elif interval_pct <= MAX_INTERVAL_PCT:
        scores["interval"] = 10
    else:
        scores["interval"] = 0

    # 3. Согласие индикаторов с прогнозом (25 очков)
    bullish = final_move_pct > 0
    indicator_agree = 0
    if bullish:
        if rsi < 50:            indicator_agree += 1   # перепроданность → рост
        if macd_cross == "bullish":  indicator_agree += 1
        if bb_zone in ("oversold", "near_lower"):  indicator_agree += 1
        if volume_trend == "растущий": indicator_agree += 1
    else:
        if rsi > 50:            indicator_agree += 1
        if macd_cross == "bearish":  indicator_agree += 1
        if bb_zone in ("overbought", "near_upper"):  indicator_agree += 1
        if volume_trend == "падающий": indicator_agree += 1

    scores["indicators"] = round(indicator_agree / 4 * 25)

    # 4. Сила тренда ADX (15 очков) — торгуем только при наличии тренда
    if adx >= 25:
        scores["adx"] = 15
    elif adx >= 20:
        scores["adx"] = 8
    else:
        scores["adx"] = 0   # флэт — не торгуем

    # 5. Стабильность ensemble (10 очков)
    if uncertainty < 0.001:
        scores["ensemble"] = 10
    elif uncertainty < 0.003:
        scores["ensemble"] = 5
    else:
        scores["ensemble"] = 0

    total = sum(scores.values())
    # Применяем вес таймфрейма (не меняем score, используем в консенсусе)
    scores["tf_weight"] = tf_weight
    scores["total"]     = total
    return total, scores


def compute_sl_tp(pred_prices: np.ndarray, q_low: np.ndarray, q_high: np.ndarray,
                  last_price: float, atr: float, is_long: bool
                  ) -> Tuple[float, float, float]:
    """
    Вычисляет стоп-лосс и тейк-профит.
    SL = на основе ATR (1.5x) и нижнего квантиля
    TP = на основе верхнего квантиля и цели прогноза
    """
    target = pred_prices[-1]
    if is_long:
        sl_atr     = last_price - 1.5 * atr
        sl_quantile = q_low[0]                       # нижняя граница первого шага
        sl         = max(sl_atr, sl_quantile)        # берём дальше от цены (осторожнее)
        tp_target  = target
        tp_quantile = q_high[-1]
        tp         = min(tp_target, tp_quantile)     # берём ближе (консервативнее)
    else:
        sl_atr     = last_price + 1.5 * atr
        sl_quantile = q_high[0]
        sl         = min(sl_atr, sl_quantile)
        tp_target  = target
        tp_quantile = q_low[-1]
        tp         = max(tp_target, tp_quantile)

    rr = abs(tp - last_price) / abs(sl - last_price) if abs(sl - last_price) > 0 else 0
    return round(sl, 2), round(tp, 2), round(rr, 2)


def is_session_active() -> Tuple[bool, str]:
    """Проверяем, находимся ли мы в активной торговой сессии MOEX"""
    now = datetime.now().time()
    if now < SESSION_START:
        return False, f"Сессия ещё не открылась (старт в {SESSION_START})"
    if now > SESSION_END:
        return False, f"Сессия закрывается (конец в {SESSION_END})"
    return True, "Сессия активна"

# ======================== LLM ========================

def query_llm(prompt: str, language: str = "russian") -> Optional[str]:
    """
    Запрос к локальной LLM.
    
    Args:
        prompt: Текст запроса
        language: Язык ответа ('russian', 'english')
    
    Returns:
        Ответ LLM или None при ошибке
    """
    if not LLM_ENABLED:
        return None
    
    # Добавляем инструкцию о языке в начало промпта
    lang_instruction = {
        "russian": "Отвечай ТОЛЬКО на русском языке.",
        "english": "Answer ONLY in English."
    }
    
    full_prompt = f"{lang_instruction.get(language, lang_instruction['russian'])}\n\n{prompt}"
    
    try:
        r = requests.post(LLM_API_URL,
                          json={"prompt": full_prompt, "n_predict": 400,
                                "temperature": 0.2, "stop": ["</s>", "###"]},
                          timeout=30)
        r.raise_for_status()
        return r.json().get("content", "").strip()
    except Exception:
        return None

def make_signal_prompt(ticker: str, signal: str, confidence: int,
                       tf_breakdown: Dict, sl: float, tp: float, rr: float,
                       indicators: Dict) -> str:
    return (
        f"Ты — трейдинговый советник. Дай краткое заключение (2-3 предложения) "
        f"о сигнале для интрадей-торговли {ticker}.\n\n"
        f"Сигнал: {signal} | Уверенность: {confidence}/100\n"
        f"Стоп-лосс: {sl} | Тейк-профит: {tp} | R/R: {rr}\n"
        f"RSI: {indicators.get('rsi')} | ADX: {indicators.get('adx')} | "
        f"Объём: {indicators.get('volume')}\n"
        f"Таймфреймы: {tf_breakdown}\n\n"
        "Ответь на русском: стоит ли входить в сделку и почему."
    )

# ======================== СОХРАНЕНИЕ ========================

def save_signal(signal_data: dict, timestamp: str):
    path = os.path.join(RESULTS_DIR, f"signal_{timestamp.replace(':', '-')}.csv")
    pd.DataFrame([signal_data]).to_csv(path, index=False, encoding="utf-8")
    return path

# ======================== ОСНОВНОЙ ПАЙПЛАЙН ========================

def main(ticker: str = None, return_signal: bool = False, logger: logging.Logger = None) -> Optional[dict]:
    """
    Основной пайплайн генерации торгового сигнала.
    
    Args:
        ticker: Тикер инструмента (по умолчанию используется глобальный TICKER)
        return_signal: Если True, возвращает dict сигнала вместо логирования
        logger: Логгер для вывода (по умолчанию создаётся новый)
    
    Returns:
        dict с сигналом если return_signal=True, иначе None
    """
    global TICKER
    
    # Используем переданный тикер или глобальный
    if ticker:
        TICKER = ticker
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Создаём логгер если не передан
    if logger is None:
        logger = setup_logger()
        print_header = True
    else:
        print_header = False
    
    if print_header:
        print(f"\n{'='*60}")
        print(f"  ТОРГОВЫЙ СИГНАЛ {TICKER} | {timestamp}")
        print(f"{'='*60}\n")

    # Проверка сессии
    session_ok, session_msg = is_session_active()
    logger.info(f"Сессия: {session_msg}")
    if not session_ok:
        logger.warning("Вне торговой сессии — сигнал информационный, не для исполнения")

    # Загрузка чекпоинта
    try:
        ckpt_path = get_checkpoint_path(LAG_LLAMA_CHECKPOINT_PATH)
        logger.info(f"Чекпоинт загружен")
        load_checkpoint_data(ckpt_path)  # предзагрузка в кэш
    except Exception as e:
        logger.error(f"Ошибка чекпоинта: {e}")
        return

    # Загрузка данных
    data_dict = load_data_tf(TICKER, START_DATE, END_DATE, logger)
    if not data_dict:
        logger.error("Нет данных")
        return

    tf_signals   = {}   # {tf_name: {"direction", "score", "pred", "sl", "tp", "rr"}}
    tf_indicators= {}   # для LLM

    for tf_name, tf_code in TIMEFRAMES.items():
        logger.info(f"\n{'─'*50}")
        logger.info(f"[ {tf_name} ]")

        if tf_name not in data_dict or len(data_dict[tf_name]) < 50:
            logger.warning(f"Нет данных для {tf_name}")
            continue

        df = data_dict[tf_name]

        # Индикаторы
        series       = prepare_log_returns(df, "Close")
        last_price   = float(df["Close"].iloc[-1])
        rsi          = compute_rsi(df["Close"])
        macd, msig, mcross = compute_macd(df["Close"])
        bb_u, bb_l, bb_pos, bb_zone = compute_bb(df["Close"])
        atr          = compute_atr(df)
        adx          = compute_adx(df)
        vol_trend    = get_volume_trend(df)
        support, res = detect_support_resistance(df)

        logger.info(
            f"Цена: {last_price:.2f} | RSI: {rsi} | ADX: {adx} | ATR: {atr:.3f}"
        )
        logger.info(
            f"MACD: {macd:.4f}/{msig:.4f} ({mcross}) | BB: {bb_zone} ({bb_pos:.0%})"
        )
        logger.info(f"Поддержка: {support:.2f} | Сопротивление: {res:.2f} | Объём: {vol_trend}")

        tf_indicators[tf_name] = dict(
            rsi=rsi, macd=macd, macd_cross=mcross,
            bb_zone=bb_zone, adx=adx, volume=vol_trend, atr=atr
        )

        # Прогноз с ensemble
        try:
            logger.info(f"Прогноз (ensemble x{ENSEMBLE_RUNS})...")
            mean_pred, (q10, q90), uncertainty = predict_ensemble(
                ckpt_path, series, tf_code,
                PREDICTION_STEPS, CONTEXT_LENGTH,
                NUM_SAMPLES, ENSEMBLE_RUNS
            )

            pred_prices  = denormalize(mean_pred, last_price)
            q_low_prices = denormalize(q10, last_price)
            q_hi_prices  = denormalize(q90, last_price)

            # Таблица прогноза
            logger.info(f"{'Шаг':<4} | {'Цель':<8} | {'Δ%':<7} | {'Q10':<8} | {'Q90':<8} | {'Ширина':<7}")
            logger.info("─" * 52)
            for i in range(PREDICTION_STEPS):
                chg   = (pred_prices[i] / last_price - 1) * 100
                width = q_hi_prices[i] - q_low_prices[i]
                logger.info(
                    f"{i+1:<4} | {pred_prices[i]:<8.2f} | {chg:+.2f}%  | "
                    f"{q_low_prices[i]:<8.2f} | {q_hi_prices[i]:<8.2f} | {width:.3f}"
                )
            logger.info(f"Нестабильность ensemble: {uncertainty:.5f}")

            # Confidence score
            final_move = (pred_prices[-1] / last_price - 1) * 100
            is_long    = final_move > 0
            score, breakdown = compute_confidence_score(
                pred_prices, q_low_prices, q_hi_prices, last_price,
                rsi, mcross, bb_zone, adx, vol_trend, uncertainty,
                TF_WEIGHTS.get(tf_name, 0.33)
            )

            # SL/TP
            sl, tp, rr = compute_sl_tp(
                pred_prices, q_low_prices, q_hi_prices, last_price, atr, is_long
            )

            direction = "LONG" if is_long else "SHORT"
            logger.info(
                f"\nСигнал: {direction} | Score: {score}/100 | "
                f"Δ: {final_move:+.2f}% | SL: {sl} | TP: {tp} | R/R: {rr}"
            )
            logger.info(f"Компоненты: {breakdown}")

            tf_signals[tf_name] = {
                "direction": direction,
                "score":     score,
                "weight":    TF_WEIGHTS.get(tf_name, 0.33),
                "move_pct":  final_move,
                "sl":        sl,
                "tp":        tp,
                "rr":        rr,
                "pred":      pred_prices[-1],
                "uncertainty": uncertainty
            }

        except Exception as e:
            logger.error(f"Ошибка прогноза [{tf_name}]: {e}")
            logger.debug(traceback.format_exc())
            continue

    # ======================== КОНСЕНСУС ========================
    logger.info(f"\n{'='*60}")
    logger.info("ИТОГОВЫЙ ТОРГОВЫЙ СИГНАЛ")
    logger.info("="*60)

    if not tf_signals:
        logger.error("Нет данных для генерации сигнала")
        return

    # Взвешенный score
    total_weight    = sum(s["weight"] for s in tf_signals.values())
    weighted_score  = sum(
        s["score"] * s["weight"] for s in tf_signals.values()
    ) / total_weight

    # Взвешенное направление
    long_weight  = sum(s["weight"] for s in tf_signals.values() if s["direction"] == "LONG")
    short_weight = sum(s["weight"] for s in tf_signals.values() if s["direction"] == "SHORT")
    consensus_dir = "LONG" if long_weight >= short_weight else "SHORT"

    # Проверяем единогласие (все таймфреймы одного направления)
    all_same = len(set(s["direction"] for s in tf_signals.values())) == 1

    # Финальные SL/TP (от 10T если есть, иначе 1H)
    ref_tf = "10T" if "10T" in tf_signals else ("1H" if "1H" in tf_signals else list(tf_signals.keys())[0])
    final_sl = tf_signals[ref_tf]["sl"]
    final_tp = tf_signals[ref_tf]["tp"]
    final_rr = tf_signals[ref_tf]["rr"]

    # Проверка R/R
    rr_ok = final_rr >= MIN_RR_RATIO
    
    # Проверка минимального прогнозного profit (не менее 0.9%)
    ref_signal = tf_signals[ref_tf]
    predicted_profit_pct = abs(ref_signal["move_pct"])
    min_profit_threshold = 0.9  # Минимальный прогноз profit в %
    profit_ok = predicted_profit_pct >= min_profit_threshold

    # Финальный вердикт
    confidence = round(weighted_score)
    signal_ok  = (confidence >= MIN_CONFIDENCE) and rr_ok and session_ok and profit_ok

    # Определяем чистое направление для заявки
    if not signal_ok:
        signal_direction = "НЕТ СИГНАЛА"
        reasons = []
        if confidence < MIN_CONFIDENCE:
            reasons.append(f"score {confidence} < порог {MIN_CONFIDENCE}")
        if not rr_ok:
            reasons.append(f"R/R {final_rr} < минимум {MIN_RR_RATIO}")
        if not session_ok:
            reasons.append("вне сессии")
        if not profit_ok:
            reasons.append(f"прогнозный profit {predicted_profit_pct:.2f}% < порог {min_profit_threshold}%")
        signal_text = f"НЕТ СИГНАЛА ({'; '.join(reasons)})"
    else:
        signal_direction = consensus_dir  # Чистое направление: LONG или SHORT
        agreement = "✓ все ТФ согласны" if all_same else "⚠ ТФ расходятся"
        signal_text = f"{consensus_dir} | {agreement}"

    logger.info(f"\n  Направление:     {consensus_dir}")
    logger.info(f"  Уверенность:     {confidence}/100")
    logger.info(f"  Стоп-лосс:       {final_sl}")
    logger.info(f"  Тейк-профит:     {final_tp}")
    logger.info(f"  Risk/Reward:     {final_rr}")
    logger.info(f"  Прогноз profit:  {predicted_profit_pct:+.2f}% (мин. {min_profit_threshold}%)")
    logger.info(f"  Сессия:          {'активна' if session_ok else 'закрыта'}")
    logger.info(f"\n>>> {signal_text} <<<")

    # Детализация по таймфреймам
    logger.info(f"\nДеталь по ТФ (вес | score | направление | R/R):")
    for tf, s in tf_signals.items():
        logger.info(
            f"  {tf:>4}: вес={s['weight']:.2f} | "
            f"score={s['score']:>3}/100 | {s['direction']:>5} | "
            f"Δ{s['move_pct']:+.2f}% | R/R={s['rr']}"
        )

    # LLM комментарий
    if LLM_ENABLED and signal_ok:
        tf_summary = {tf: f"{s['direction']} ({s['score']})" for tf, s in tf_signals.items()}
        indic_summary = {
            "rsi":    tf_indicators.get("10T", {}).get("rsi", "—"),
            "adx":    tf_indicators.get("10T", {}).get("adx", "—"),
            "volume": tf_indicators.get("10T", {}).get("volume", "—"),
        }
        prompt = make_signal_prompt(
            TICKER, signal_text, confidence,
            tf_summary, final_sl, final_tp, final_rr, indic_summary
        )
        llm_resp = query_llm(prompt)
        if llm_resp:
            logger.info(f"\nАНАЛИЗ LLM:\n{llm_resp}")

    # Сохранение сигнала
    signal_data = {
        "ticker":      TICKER,
        "signal":      signal_text,
        "confidence":  confidence,
        "sl":          final_sl,
        "tp":          final_tp,
        "rr":          final_rr,
        "session_ok":  session_ok,
        "timestamp":   timestamp,
        "predicted_profit_pct": predicted_profit_pct,
        **{f"{tf}_score": s["score"] for tf, s in tf_signals.items()},
        **{f"{tf}_dir":   s["direction"] for tf, s in tf_signals.items()},
    }
    csv_path = save_signal(signal_data, timestamp)
    logger.info(f"\nСигнал сохранён: {csv_path}")
    logger.info(f"Лог: signal_log.txt")
    logger.info("="*60)
    
    # Возвращаем сигнал если запрошено
    if return_signal:
        # Формируем упрощённый dict для возврата с чистым направлением
        result = {
            "ticker": TICKER,
            "signal": signal_direction,  # Чистое направление: LONG, SHORT или НЕТ СИГНАЛА
            "confidence": confidence,
            "sl": final_sl,
            "tp": final_tp,
            "rr": final_rr,
            "session_ok": session_ok,
            "timestamp": timestamp,
            "predicted_profit_pct": predicted_profit_pct
        }
        return result
    
    return None


if __name__ == "__main__":
    main()