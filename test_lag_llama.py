#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прогнозирование движений SNGS на 3 таймфреймах с использованием:
- Lag-Llama (https://github.com/time-series-foundation-models/lag-llama) для forecasting
- Локального LLM-сервера (127.0.0.1:8080) для интерпретации результатов

Запуск:
    python predict_sngs.py
"""

import os
import sys
import json
import time
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import requests
from tqdm import tqdm

# Lag-Llama imports
try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from lag_llama.gluon.lightning_module import LagLlamaLightningModule
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.torch.distributions.studentT import StudentTOutput
    from gluonts.torch.modules.loss import NegativeLogLikelihood
except ImportError:
    print("❌ Lag-Llama не установлен. Выполните: pip install git+https://github.com/time-series-foundation-models/lag-llama.git")
    sys.exit(1)

# Ваши импорты
import moexalgo
import vectorbt as vbt

warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)
np.random.seed(42)
torch.manual_seed(42)

# ======================== 🔧 НАСТРОЙКИ ========================
DEBUG = True
CACHE_DIR = 'cache_data'
os.makedirs(CACHE_DIR, exist_ok=True)

# Тикер и таймфреймы
TICKER = 'SNGS'
TIMEFRAMES = {
    '1D': '1d',
    '1H': '1h',
    '10T': '10min'
}

# Период данных
START_DATE = pd.Timestamp('2025-01-01')
END_DATE = pd.Timestamp('2026-05-01')

# Параметры прогноза
PREDICTION_STEPS = 5  # Сколько шагов вперёд предсказывать
CONTEXT_LENGTH = 200  # Длина контекста для Lag-Llama

# Локальный LLM сервер
LLM_API_URL = "http://127.0.0.1:8080/completion"
LLM_ENABLED = True  # Отключите, если сервер не запущен

# Параметры Lag-Llama
LAG_LLAMA_MODEL_PATH = None  # None = использовать предобученную модель
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Используемое устройство: {DEVICE}")


# ======================== 📥 ЗАГРУЗКА ДАННЫХ ========================

def load_data_tf(ticker_name: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[Dict[str, pd.DataFrame]]:
    """Загружает OHLCV данные с MOEX с чанкованием и кэшированием"""
    tf_map = {'1D': '1d', '1H': '1h', '10T': '10min'}

    # Разбиваем период на чанки по 30 дней
    date_ranges = pd.date_range(start=start, end=end, freq='30D')
    ranges = [(date_ranges[i], date_ranges[i + 1]) for i in range(len(date_ranges) - 1)]
    if date_ranges[-1] < end:
        ranges.append((date_ranges[-1], end))

    dataframes = {}

    for tf_name, tf_code in tf_map.items():
        cache_file = os.path.join(CACHE_DIR,
                                  f"{ticker_name}_{tf_name}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")

        # Проверка кэша
        if os.path.exists(cache_file):
            print(f"📂 [{tf_name}] Загружаем из кэша...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                dataframes[tf_name] = df
                continue

        print(f"📥 [{tf_name}] Загружаем {ticker_name} с MOEX ({tf_code})...")
        ticker = moexalgo.Ticker(ticker_name)
        chunk_dfs = []

        for chunk_start, chunk_end in ranges:
            try:
                # Для MOEX Algo период должен быть в формате: '1m', '5m', '10m', '15m', '30m', '1h', '1d'
                period_param = tf_code
                
                df_chunk = ticker.candles(
                    start=chunk_start.strftime('%Y-%m-%d'),
                    end=chunk_end.strftime('%Y-%m-%d'),
                    period=period_param
                )
                if df_chunk is not None and not df_chunk.empty:
                    # Нормализация колонок
                    idx_col = 'date' if 'date' in df_chunk.columns else 'begin'
                    df_chunk = df_chunk.set_index(idx_col)
                    df_chunk = df_chunk.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    })
                    df_chunk.index = pd.to_datetime(df_chunk.index)
                    chunk_dfs.append(df_chunk)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки чанка [{tf_name}] {chunk_start}: {e}")
                continue

        if not chunk_dfs:
            print(f"❌ [{tf_name}] Не удалось получить данные.")
            continue

        # Объединение и очистка
        df = pd.concat(chunk_dfs).sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Сохранение в кэш
        df.to_csv(cache_file)
        print(f"💾 [{tf_name}] Сохранено в кэш: {len(df)} баров")

        # Фильтрация колонок и обработка пропусков
        needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[needed].dropna()
        for col in ['Close', 'High', 'Low']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill().clip(lower=0.01)

        dataframes[tf_name] = df

    return dataframes if dataframes else None


# ======================== 🔄 ПРЕДОБРАБОТКА ДЛЯ LAG-LLAMA ========================

def prepare_multivariate_series(df: pd.DataFrame, target_col: str = 'Close') -> pd.Series:
    """
    Подготавливает multivariate OHLCV данные для Lag-Llama.
    Lag-Llama работает с univariate, поэтому используем основной таргет (Close),
    но можно добавить фичи через кастомный энкодер (расширение).
    """
    # Нормализация через log-returns для стационарности
    series = df[target_col].copy()

    # Логарифмические возвраты (уменьшает нестационарность)
    log_returns = np.log(series / series.shift(1)).dropna()

    # Заполнение пропусков
    log_returns = log_returns.ffill().bfill()

    return log_returns


def create_gluonts_dataset(series: pd.Series, freq: str) -> PandasDataset:
    """Создаёт GluonTS Dataset из pandas Series"""
    # GluonTS ожидает данные в специфичном формате
    dataset = [{
        "start": series.index[0],
        "target": series.values,
        "feat_static_cat": [0],  # dummy feature
        "item_id": "SNGS"
    }]
    return PandasDataset(dataset, freq=freq)


# ======================== 🔮 ПРОГНОЗИРОВАНИЕ С LAG-LLAMA ========================

def load_lag_llama_model(model_path: Optional[str] = None):
    """Загружает предобученную модель Lag-Llama"""
    print(f"🤖 Загрузка Lag-Llama модели на {DEVICE}...")

    # Добавляем необходимые классы в безопасные глобальные объекты для torch.load
    torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])
    
    # Lag-Llama использует хаб моделей HuggingFace
    if model_path is None:
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id="time-series-foundation-models/Lag-Llama",
                filename="lag-llama.ckpt",
                repo_type="model"
            )
            print(f"📥 Модель загружена с HuggingFace: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели с HuggingFace: {e}")
            raise
    
    # Загружаем модель из чекпоинта
    model = LagLlamaLightningModule.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    print("✅ Модель успешно загружена!")
    return model


def predict_with_lag_llama(
        model,
        series: pd.Series,
        freq: str,
        prediction_steps: int,
        context_length: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Делает прогноз на prediction_steps шагов вперёд с использованием Lag-Llama.
    Возвращает: (mean_prediction, quantiles_0.1_0.9)
    """
    from gluonts.dataset.field_names import FieldName
    from gluonts.transform import ExpectedNumInstanceSampler, InstanceSplitter
    
    # Убеждаемся, что у нас достаточно данных
    if len(series) < context_length:
        # Если данных мало, используем все доступные и уменьшаем context_length
        actual_context = max(len(series), 10)  # минимум 10 точек
        print(f"⚠️ Мало данных ({len(series)}), используем контекст {actual_context}")
    else:
        actual_context = context_length
    
    # Берём последние actual_context точек
    series_short = series.tail(actual_context)
    
    # Подготовка данных для модели
    data_entry = {
        "start": series_short.index[0],
        "target": series_short.values.astype(float),
        "item_id": "SNGS"
    }
    
    # Создаем трансформацию для подготовки данных
    # Используем InstanceSplitter с TestSplitSampler для инференса
    from gluonts.transform import InstanceSplitter, TestSplitSampler
    
    transformation = InstanceSplitter(
        target_field='target',
        is_pad_field='is_pad',
        start_field='start',
        forecast_start_field='forecast_start',
        instance_sampler=TestSplitSampler(min_past=actual_context),
        past_length=actual_context,
        future_length=prediction_steps
    )
    
    # Применяем трансформацию
    transformed_data = list(transformation.apply(iter([data_entry])))
    
    if not transformed_data:
        # Фолбэк: пробуем создать входные данные вручную
        print("⚠️ TestSplitter не вернул данных, пробуем ручной формат...")
        target_values = series_short.values.astype(float)
        input_tensor = torch.tensor(target_values).unsqueeze(0).unsqueeze(-1).float().to(DEVICE)
    else:
        # Берём первый батч
        batch = transformed_data[0]
        if 'past_target' in batch:
            input_tensor = torch.tensor(batch['past_target']).unsqueeze(0).float().to(DEVICE)
        elif 'target' in batch:
            # Альтернативный формат
            target_data = batch['target']
            if isinstance(target_data, dict) and 'data' in target_data:
                target_data = target_data['data']
            input_tensor = torch.tensor(target_data).unsqueeze(0).float().to(DEVICE)
        else:
            raise ValueError("Не удалось подготовить данные для прогнозирования")
    
    # Устанавливаем модель в режим eval
    model.eval()
    model = model.to(DEVICE)
    
    # Генерируем прогноз
    with torch.no_grad():
        # Создаем тензор наблюдаемых значений (все значения наблюдаемы)
        past_observed_values = torch.ones_like(input_tensor).to(DEVICE)
        
        # Получаем параметры распределения от модели
        params = model.model(input_tensor, past_observed_values=past_observed_values)
        
        # Используем StudentT распределение для семплирования
        distr_output = model.distr_output
        distr = distr_output.distribution(*params, scale=1.0)
        
        # Семплируем прогнозы
        samples = distr.sample(sample_shape=(100,))  # 100 семплов
        
        # Вычисляем среднее и квантили
        mean_pred = samples.mean(dim=0).cpu().numpy()
        q_low = samples.quantile(0.1, dim=0).cpu().numpy()
        q_high = samples.quantile(0.9, dim=0).cpu().numpy()
    
    return mean_pred, (q_low, q_high)


# ======================== 🤖 ИНТЕГРАЦИЯ С ЛОКАЛЬНЫМ LLM ========================

def query_local_llm(prompt: str, max_tokens: int = 512) -> Optional[str]:
    """Отправляет промпт на локальный LLM-сервер (llama.cpp / koboldcpp API)"""
    if not LLM_ENABLED:
        return None

    try:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.3,
            "stop": ["</s>", "###", "\n\n"]
        }
        response = requests.post(LLM_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("content", "").strip()
    except Exception as e:
        print(f"⚠️ Ошибка запроса к LLM: {e}")
        return None


def generate_analysis_prompt(
        ticker: str,
        tf: str,
        last_prices: List[float],
        predictions: np.ndarray,
        quantiles: Tuple[float, float],
        volume_trend: str
) -> str:
    """Формирует промпт для анализа прогноза"""
    pred_list = predictions.tolist()
    q_low, q_high = quantiles

    prompt = f"""Ты — финансовый аналитик. Проанализируй прогноз для {ticker} на таймфрейме {tf}.

📊 Последние 5 цен закрытия: {[round(p, 2) for p in last_prices]}
🔮 Прогноз на 5 шагов вперёд (среднее): {[round(p, 4) for p in pred_list]}
📏 Доверительный интервал (10%-90%): [{round(q_low, 4)}, {round(q_high, 4)}]
📈 Тренд объёма: {volume_trend}

Дай краткий вывод:
1. Ожидаемое направление (бычий/медвежий/боковик)
2. Уровень уверенности (низкий/средний/высокий)
3. Ключевой риск или возможность

Ответь на русском, 3-4 предложения."""

    return prompt


# ======================== 📊 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========================

def get_volume_trend(df: pd.DataFrame, window: int = 20) -> str:
    """Определяет тренд объёма"""
    recent = df['Volume'].iloc[-window:].mean()
    older = df['Volume'].iloc[-2 * window:-window].mean()
    if recent > older * 1.2:
        return "растущий 📈"
    elif recent < older * 0.8:
        return "падающий 📉"
    return "нейтральный ➡️"


def denormalize_predictions(preds: np.ndarray, last_price: float) -> np.ndarray:
    """Возвращает лог-возвраты к абсолютным ценам"""
    # cumulative sum для восстановления уровня цен
    cumulative = np.cumsum(np.concatenate([[0], preds]))
    return last_price * np.exp(cumulative)[1:]  # пропускаем первый 0


# ======================== 🚀 ОСНОВНОЙ ПЛАЙПЛАЙН ========================

def main():
    print(f"🚀 Запуск прогноза для {TICKER} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # 1. Загрузка данных
    data_dict = load_data_tf(TICKER, START_DATE, END_DATE)
    if not data_dict:
        print("❌ Критическая ошибка: не удалось загрузить данные.")
        return

    # 2. Загрузка модели
    try:
        estimator = load_lag_llama_model(LAG_LLAMA_MODEL_PATH)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # Настройка логирования
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('prediction_log.txt', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Запуск прогноза для {TICKER}")

    # 3. Прогнозирование по каждому таймфрейму
    results = {}
    
    for tf_name, tf_code in TIMEFRAMES.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 Обработка таймфрейма: {tf_name}")
        
        df = data_dict[tf_name]
        if df.empty or len(df) < CONTEXT_LENGTH:
            logger.warning(f"⚠️ Недостаточно данных для {tf_name} (нужно минимум {CONTEXT_LENGTH})")
            continue
        
        # Подготовка данных
        series = prepare_multivariate_series(df, 'Close')
        last_price = df['Close'].iloc[-1]
        last_prices = df['Close'].iloc[-5:].tolist()
        volume_trend = get_volume_trend(df)
        
        logger.info(f"💹 Последняя цена: {last_price:.2f}")
        logger.info(f"📈 Тренд объёма: {volume_trend}")
        
        # Прогноз Lag-Llama (на лог-возвратах)
        try:
            mean_pred, quantiles = predict_with_lag_llama(
                model=estimator,
                series=series,
                freq=tf_code,
                prediction_steps=PREDICTION_STEPS,
                context_length=CONTEXT_LENGTH
            )
            
            # Денормализация к абсолютным ценам
            pred_prices = denormalize_predictions(mean_pred, last_price)
            q_low_prices = denormalize_predictions(np.array([quantiles[0]] * PREDICTION_STEPS), last_price)
            q_high_prices = denormalize_predictions(np.array([quantiles[1]] * PREDICTION_STEPS), last_price)
            
            # Вывод прогноза
            logger.info(f"\n🔮 ПРОГНОЗ НА {PREDICTION_STEPS} ШАГОВ ({tf_name}):")
            logger.info(f"{'Шаг':<6} | {'Прогноз':<10} | {'Низ (10%)':<10} | {'Верх (90%)':<10}")
            logger.info("-" * 45)
            
            for i in range(PREDICTION_STEPS):
                logger.info(
                    f"{i+1:<6} | {pred_prices[i]:<10.2f} | {q_low_prices[i]:<10.2f} | {q_high_prices[i]:<10.2f}"
                )
            
            # Сохранение результата
            results[tf_name] = {
                'predictions': pred_prices,
                'quantile_low': q_low_prices,
                'quantile_high': q_high_prices,
                'last_price': last_price
            }
            
            # Анализ через LLM (если доступен)
            if LLM_ENABLED:
                logger.info("\n🤖 Запрос анализа у локальной LLM...")
                prompt = generate_analysis_prompt(
                    ticker=TICKER,
                    tf=tf_name,
                    last_prices=last_prices,
                    predictions=pred_prices,
                    quantiles=(q_low_prices[-1], q_high_prices[-1]),
                    volume_trend=volume_trend
                )
                
                llm_response = query_local_llm(prompt)
                if llm_response:
                    logger.info(f"\n💬 АНАЛИЗ LLM:\n{llm_response}")
                else:
                    logger.info("⚠️ LLM не ответил (возможно сервер недоступен)")
                    
        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования для {tf_name}: {e}")
            logger.error(traceback.format_exc())
            continue

    # Итоговый вывод
    logger.info(f"\n{'='*50}")
    logger.info("✅ ПРОГНОЗИРОВАНИЕ ЗАВЕРШЕНО")
    logger.info(f"📁 Лог сохранён в: prediction_log.txt")
    
    if results:
        logger.info("\n📋 СВОДКА ПРОГНОЗОВ:")
        for tf_name, res in results.items():
            direction = "📈 рост" if res['predictions'][-1] > res['last_price'] else "📉 падение"
            change_pct = ((res['predictions'][-1] / res['last_price']) - 1) * 100
            logger.info(f"{tf_name}: {direction} ({change_pct:+.2f}%) к последнему шагу")


if __name__ == "__main__":
    main()