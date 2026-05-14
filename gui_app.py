#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui_app.py — Графический интерфейс для системы торговли Lag-Llama + T-Bank

Функционал:
- Ввод тикера и параметров торговли
- Отображение сигналов от модели Lag-Llama
- Визуализация confidence, SL/TP, R/R ratio
- Выставление заявок через T-Invest API
- Логирование операций в реальном времени
"""

import os
import sys
import logging
import threading
import queue
from datetime import datetime
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# Матplotlib для графиков
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# Импорт основной логики
import test_lag_llama
from main import (
    generate_signal,
    validate_signal,
    get_figi_with_moex_store,
    place_order,
    setup_logger as setup_main_logger,
    MIN_CONFIDENCE,
    MIN_RR_RATIO,
    MIN_PROFIT_PCT
)


class TradingGUI:
    """Основной класс графического интерфейса"""
    
    def __init__(self, root: tk.Tk):
        """
        Инициализация GUI
        
        Args:
            root: Главный окно tkinter
        """
        self.root = root
        self.root.title("Lag-Llama Trading System v2.0")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Очередь для безопасного обновления UI из потоков
        self.message_queue = queue.Queue()
        
        # Переменные состояния
        self.is_trading = False
        self.current_signal: Optional[Dict[str, Any]] = None
        self.token: str = ""
        self.account_id: str = ""
        
        # Настройка стилей
        self._setup_styles()
        
        # Создание интерфейса
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        
        # Запуск обработки очереди сообщений
        self._process_queue()
        
        # Логирование
        self.logger = self._setup_gui_logger()
        
        self.logger.info("GUI запущен. Ожидание ввода тикера...")
    
    def _setup_styles(self):
        """Настройка стилей виджетов"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Цветовая схема
        self.colors = {
            'bg': '#f0f0f0',
            'panel_bg': '#ffffff',
            'accent': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'text': '#333333'
        }
        
        # Настройка шрифтов
        self.fonts = {
            'title': ('Arial', 14, 'bold'),
            'normal': ('Arial', 10),
            'bold': ('Arial', 10, 'bold'),
            'small': ('Arial', 9),
            'console': ('Consolas', 9)
        }
    
    def _create_menu(self):
        """Создание меню приложения"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Сохранить лог", command=self._save_log)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self._on_closing)
        
        # Меню Настройки
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(label="Параметры торговли", command=self._show_trade_settings)
        settings_menu.add_command(label="API Токен", command=self._show_token_settings)
        
        # Меню Справка
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе", command=self._show_about)
        help_menu.add_command(label="Документация", command=self._show_docs)
    
    def _create_main_layout(self):
        """Создание основной компоновки интерфейса"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Левая панель - управление
        left_panel = ttk.LabelFrame(main_frame, text="Управление торговлей", padding="10")
        left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        
        # Правая панель - информация и логи
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Создание виджетов левой панели
        self._create_ticker_input(left_panel)
        self._create_signal_button(left_panel)
        self._create_trade_button(left_panel)
        self._create_settings_frame(left_panel)
        
        # Создание виджетов правой панели
        self._create_chart_area(right_panel)  # График сверху
        self._create_signal_display(right_panel)  # Параметры сигнала
        self._create_log_console(right_panel)  # Логи снизу
    
    def _create_ticker_input(self, parent):
        """Создание поля ввода тикера"""
        ticker_frame = ttk.LabelFrame(parent, text="Тикер инструмента", padding="10")
        ticker_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ticker_frame.columnconfigure(1, weight=1)
        
        ttk.Label(ticker_frame, text="Тикер:").grid(row=0, column=0, sticky="w", pady=5)
        self.ticker_var = tk.StringVar(value="SNGS")
        self.ticker_entry = ttk.Entry(ticker_frame, textvariable=self.ticker_var, font=self.fonts['bold'])
        self.ticker_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        ttk.Label(ticker_frame, text="Примеры: SBER, GAZP, LKOH", 
                 font=self.fonts['small'], foreground='gray').grid(row=1, column=0, columnspan=2, sticky="w")
    
    def _create_signal_button(self, parent):
        """Создание кнопки получения сигнала"""
        self.signal_btn = ttk.Button(
            parent, 
            text="📊 Получить сигнал",
            command=self._get_signal_threaded,
            style='Accent.TButton'
        )
        self.signal_btn.grid(row=1, column=0, sticky="ew", pady=10)
    
    def _create_trade_button(self, parent):
        """Создание кнопки торговли"""
        self.trade_btn = ttk.Button(
            parent,
            text="💰 Выставить заявку",
            command=self._execute_trade_threaded,
            state='disabled'
        )
        self.trade_btn.grid(row=2, column=0, sticky="ew", pady=10)
    
    def _create_settings_frame(self, parent):
        """Создание рамки настроек"""
        settings_frame = ttk.LabelFrame(parent, text="Параметры сигнала", padding="10")
        settings_frame.grid(row=3, column=0, sticky="ew", pady=10)
        settings_frame.columnconfigure(1, weight=1)
        
        # Минимальный confidence
        ttk.Label(settings_frame, text="Min Confidence (%):").grid(row=0, column=0, sticky="w", pady=5)
        self.min_confidence_var = tk.IntVar(value=MIN_CONFIDENCE)
        ttk.Spinbox(settings_frame, from_=0, to=100, textvariable=self.min_confidence_var, width=10).grid(row=0, column=1, sticky="w", pady=5)
        
        # Минимальный R/R
        ttk.Label(settings_frame, text="Min R/R Ratio:").grid(row=1, column=0, sticky="w", pady=5)
        self.min_rr_var = tk.DoubleVar(value=MIN_RR_RATIO)
        ttk.Spinbox(settings_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.min_rr_var, width=10).grid(row=1, column=1, sticky="w", pady=5)
        
        # Минимальный profit
        ttk.Label(settings_frame, text="Min Profit (%):").grid(row=2, column=0, sticky="w", pady=5)
        self.min_profit_var = tk.DoubleVar(value=MIN_PROFIT_PCT)
        ttk.Spinbox(settings_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.min_profit_var, width=10).grid(row=2, column=1, sticky="w", pady=5)
    
    def _create_chart_area(self, parent):
        """Создание области для графика цены с индикаторами"""
        chart_frame = ttk.LabelFrame(parent, text="График цены и сигнал", padding="5")
        chart_frame.pack(fill="both", expand=True, padx=0, pady=(0, 10))
        
        # Создание фигуры matplotlib
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Цена инструмента", fontsize=12)
        self.ax.set_xlabel("Время", fontsize=10)
        self.ax.set_ylabel("Цена", fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
        # Встраивание графика в tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Панель инструментов (масштабирование, сохранение)
        toolbar_frame = ttk.Frame(chart_frame)
        toolbar_frame.pack(fill="x", pady=(5, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Инициализация пустых данных
        self.chart_data = None
    
    def _create_signal_display(self, parent):
        """Создание панели отображения сигнала"""
        display_frame = ttk.LabelFrame(parent, text="Текущий сигнал", padding="10")
        display_frame.pack(fill="both", expand=True, padx=0, pady=(0, 10))
        
        # Сетка для параметров сигнала
        self.signal_labels = {}
        params = [
            ("Тикер:", "ticker"),
            ("Направление:", "direction"),
            ("Confidence:", "confidence"),
            ("Цена входа:", "entry_price"),
            ("Stop Loss:", "sl"),
            ("Take Profit:", "tp"),
            ("R/R Ratio:", "rr"),
            ("Прогноз profit:", "profit_pct"),
            ("Статус:", "status")
        ]
        
        for i, (label_text, key) in enumerate(params):
            ttk.Label(display_frame, text=label_text, font=self.fonts['bold']).grid(row=i, column=0, sticky="w", pady=3, padx=5)
            value_label = ttk.Label(display_frame, text="-", font=self.fonts['normal'], width=30, anchor="w")
            value_label.grid(row=i, column=1, sticky="w", pady=3)
            self.signal_labels[key] = value_label
        
        display_frame.columnconfigure(1, weight=1)
    
    def _create_log_console(self, parent):
        """Создание консоли логов"""
        log_frame = ttk.LabelFrame(parent, text="Лог операций", padding="10")
        log_frame.pack(fill="both", expand=True)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_console = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            font=self.fonts['console'],
            wrap=tk.WORD,
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='white'
        )
        self.log_console.grid(row=0, column=0, sticky="nsew")
        
        # Контекстное меню для копирования
        self.log_console.bind("<Button-3>", self._show_copy_menu)
    
    def _create_status_bar(self):
        """Создание строки состояния"""
        self.status_bar = ttk.Label(
            self.root,
            text="Готов к работе | Сессия: закрыта",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(10, 5)
        )
        self.status_bar.grid(row=1, column=0, sticky="ew")
    
    def _setup_gui_logger(self) -> logging.Logger:
        """Настройка логгера для GUI"""
        import logging
        
        logger = logging.getLogger("gui_logger")
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            logger.handlers.clear()
        
        # Кастомный хендлер для вывода в GUI
        class GUIHandler(logging.Handler):
            def __init__(self, gui_instance):
                super().__init__()
                self.gui = gui_instance
            
            def emit(self, record):
                msg = self.format(record)
                self.gui.message_queue.put(('log', msg, record.levelno))
        
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s | %(message)s", datefmt="%H:%M:%S")
        handler = GUIHandler(self)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _process_queue(self):
        """Обработка очереди сообщений (безопасное обновление UI)"""
        try:
            while True:
                msg_type, msg_data, *extra = self.message_queue.get_nowait()
                
                if msg_type == 'log':
                    level = extra[0] if extra else logging.INFO
                    self._append_to_log(msg_data, level)
                
                elif msg_type == 'signal':
                    self._update_signal_display(msg_data)
                
                elif msg_type == 'status':
                    self.status_bar.config(text=msg_data)
                
                elif msg_type == 'enable_trade':
                    self.trade_btn.config(state='normal' if msg_data else 'disabled')
                    
        except queue.Empty:
            pass
        
        # Планируем следующую проверку через 100мс
        self.root.after(100, self._process_queue)
    
    def _append_to_log(self, message: str, level: int = logging.INFO):
        """Добавление сообщения в лог консоль"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Цвета для разных уровней
        color_map = {
            logging.DEBUG: '#808080',
            logging.INFO: '#4CAF50',
            logging.WARNING: '#FF9800',
            logging.ERROR: '#F44336'
        }
        
        color = color_map.get(level, '#d4d4d4')
        
        self.log_console.insert(tk.END, f"[{timestamp}] ", 'timestamp')
        self.log_console.insert(tk.END, f"{message}\n", color)
        self.log_console.see(tk.END)
        
        # Настройка тегов для цветов
        self.log_console.tag_config('timestamp', foreground='#808080')
        self.log_console.tag_config(color, foreground=color)
    
    def _update_signal_display(self, signal: Dict[str, Any]):
        """Обновление отображения сигнала и графика"""
        mappings = {
            'ticker': signal.get('ticker', '-'),
            'direction': signal.get('signal', '-'),
            'confidence': f"{signal.get('confidence', 0)}%",
            'entry_price': f"{signal.get('entry_price', 0):.4f}" if signal.get('entry_price') else '-',
            'sl': f"{signal.get('sl', 0):.4f}",
            'tp': f"{signal.get('tp', 0):.4f}",
            'rr': f"{signal.get('rr', 0):.2f}",
            'profit_pct': f"{signal.get('predicted_profit_pct', 0):+.2f}%",
            'status': '✅ Готов к торговле' if signal.get('valid', False) else '❌ Не соответствует критериям'
        }
        
        for key, value in mappings.items():
            self.signal_labels[key].config(text=value)
            
            # Подсветка статуса
            if key == 'status':
                if '✅' in value:
                    self.signal_labels[key].config(foreground=self.colors['success'])
                else:
                    self.signal_labels[key].config(foreground=self.colors['danger'])
        
        # Диагностика chart_data
        chart_data = signal.get('chart_data')
        if chart_data is not None:
            self.logger.info(f"chart_data получена: тип={type(chart_data)}, размер={len(chart_data) if hasattr(chart_data, '__len__') else 'N/A'}")
            if hasattr(chart_data, 'columns'):
                self.logger.info(f"Колонки chart_data: {list(chart_data.columns)}")
            # Обновление графика с данными из сигнала
            self._update_chart(signal)
        else:
            self.logger.warning("chart_data отсутствует в сигнале!")
    
    def _update_chart(self, signal: Dict[str, Any]):
        """
        Отрисовка графика: исторические свечи + прогноз с TP/SL
        
        Args:
            signal: Словарь с данными сигнала, включая chart_data (DataFrame с OHLCV)
        """
        try:
            import pandas as pd
            import matplotlib.dates as mdates
            
            chart_data = signal.get('chart_data')
            if chart_data is None or len(chart_data) == 0:
                self.logger.warning("Нет данных для отрисовки графика")
                return
            
            # Очистка и пересоздание оси
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            
            # Получение данных
            if isinstance(chart_data, pd.DataFrame):
                df = chart_data.copy()
                
                # Проверка наличия необходимых колонок для свечей
                required_cols = ['open', 'high', 'low', 'close']
                has_ohlcv = all(col in df.columns for col in required_cols)
                
                # Определение колонки времени
                time_col = 'time' if 'time' in df.columns else df.columns[0]
                
                # Преобразование времени в datetime
                if time_col:
                    x_dates = pd.to_datetime(df[time_col], errors='coerce')
                else:
                    x_dates = pd.date_range(start='2024-01-01', periods=len(df), freq='T')
                
                # Используем числовые позиции для отрисовки
                x_positions = list(range(len(df)))
                
                # --- 1. Отрисовка исторических свечей ---
                if has_ohlcv:
                    for idx, row in df.iterrows():
                        t = x_positions[idx]
                        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
                        
                        # Цвет свечи
                        color = '#26a69a' if c >= o else '#ef5350'
                        height = c - o
                        bottom = min(o, c)
                        if abs(height) < 0.0001:
                            height = 0.0001
                        
                        # Тело свечи
                        self.ax.add_patch(plt.Rectangle((t - 0.4, bottom), 0.8, height, 
                                                       facecolor=color, edgecolor='black', linewidth=0.5))
                        # Фитили
                        self.ax.plot([t, t], [l, h], color='black', linewidth=0.8)
                    
                    self.logger.info(f"Отрисовано {len(df)} свечей")
                else:
                    # Если нет OHLCV, рисуем просто линию цены
                    price_col = 'close' if 'close' in df.columns else df.columns[-1]
                    y_values = df[price_col].values
                    self.ax.plot(x_positions, y_values, 'b-', linewidth=1.5, label='Цена')
                
                # Последняя цена и позиция
                last_idx = len(df) - 1
                last_x = x_positions[last_idx]
                last_close = float(df['close'].iloc[last_idx]) if 'close' in df.columns else float(df.iloc[last_idx, -1])
                
                # Подписи оси X с датами
                step = max(1, len(x_dates) // 8)
                tick_positions = x_positions[::step]
                tick_labels = [x_dates[i].strftime('%d.%m %H:%M') if hasattr(x_dates[i], 'strftime') else str(x_dates[i]) 
                               for i in range(0, len(x_dates), step)]
                self.ax.set_xticks(tick_positions)
                self.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
                
                # --- 2. Отрисовка прогноза и уровней ---
                entry_price = float(signal.get('entry_price', last_close))
                tp = signal.get('tp')
                sl = signal.get('sl')
                direction = signal.get('signal', signal.get('direction', ''))
                confidence = float(signal.get('confidence', 0))
                
                # Прогнозируемая точка (сдвиг в будущее на 2 позиции)
                future_x = last_x + 2
                
                if tp:
                    tp = float(tp)
                    # Цвет TP
                    color_tp = '#26a69a' if direction in ['BUY', 'LONG'] else '#ef5350'
                    self.ax.axhline(y=tp, color=color_tp, linestyle='--', linewidth=2, label=f'TP: {tp:.4f}')
                    # Область между входом и TP
                    self.ax.fill_between([last_x, future_x], [entry_price, tp], alpha=0.15, color=color_tp)
                
                if sl:
                    sl = float(sl)
                    # Цвет SL
                    color_sl = '#ef5350' if direction in ['BUY', 'LONG'] else '#26a69a'
                    self.ax.axhline(y=sl, color=color_sl, linestyle='-.', linewidth=2, label=f'SL: {sl:.4f}')
                
                # Точка входа
                self.ax.scatter(last_x, entry_price, color='blue', s=150, zorder=5, marker='*', 
                               label=f'Вход: {entry_price:.4f}')
                
                # Линия прогноза (от входа к TP)
                if tp:
                    self.ax.plot([last_x, future_x], [entry_price, tp], color='blue', 
                                linewidth=2, linestyle=':', alpha=0.7)
                    # Подпись прогноза
                    label_text = f"Прогноз ({direction})\nConf: {confidence:.1f}%"
                    self.ax.text(future_x, tp, label_text, fontsize=9, ha='left', va='bottom',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue', boxstyle='round,pad=0.5'))
                
                # --- Оформление ---
                ticker = signal.get('ticker', '')
                self.ax.set_title(f"{ticker} - {direction} | Прогноз и Уровни", fontsize=12, fontweight='bold')
                self.ax.set_xlabel("Время", fontsize=10)
                self.ax.set_ylabel("Цена", fontsize=10)
                self.ax.grid(True, which='both', linestyle='--', alpha=0.5)
                self.ax.legend(loc='upper left', fontsize=9)
                
                # Автомасштабирование с отступами
                self.ax.margins(x=0.02, y=0.05)
                
                self.fig.tight_layout()
            
            # Перерисовка
            self.canvas.draw_idle()
            self.canvas.flush_events()
            
            self.logger.info(f"График обновлён: направление={direction}, TP={tp}, SL={sl}")
            
        except Exception as e:
            self.logger.error(f"Ошибка отрисовки графика: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _get_signal_threaded(self):
        """Запуск получения сигнала в отдельном потоке"""
        ticker = self.ticker_var.get().strip().upper()
        
        if not ticker:
            messagebox.showwarning("Предупреждение", "Введите тикер инструмента!")
            return
        
        def worker():
            try:
                self.message_queue.put(('status', f"Генерация сигнала для {ticker}..."))
                
                # Генерация сигнала
                signal = generate_signal(ticker, self.logger)
                
                if signal:
                    # Валидация
                    is_valid, reason = validate_signal(signal)
                    signal['valid'] = is_valid
                    
                    if not is_valid:
                        self.logger.warning(f"Сигнал отклонён: {reason}")
                    
                    self.message_queue.put(('signal', signal))
                    self.message_queue.put(('enable_trade', is_valid))
                    self.message_queue.put(('status', f"Сигнал получен | Валиден: {is_valid}"))
                else:
                    self.message_queue.put(('status', "Ошибка получения сигнала"))
                    
            except Exception as e:
                self.logger.error(f"Ошибка: {e}")
                self.message_queue.put(('status', "Ошибка генерации сигнала"))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _execute_trade_threaded(self):
        """Запуск торговли в отдельном потоке"""
        if not self.current_signal or not self.current_signal.get('valid'):
            messagebox.showwarning("Предупреждение", "Отсутствует валидный сигнал для торговли!")
            return
        
        # Проверка токена
        token = os.getenv('INVEST_TOKEN')
        if not token:
            messagebox.showerror("Ошибка", "INVEST_TOKEN не установлен!\n\n"
                                           "Установите переменную окружения:\n"
                                           "Windows: $env:INVEST_TOKEN='ваш_токен'\n"
                                           "Linux/Mac: export INVEST_TOKEN='ваш_токен'")
            return
        
        def worker():
            try:
                self.message_queue.put(('status', "Выполнение заявки..."))
                self.logger.info("Начало выполнения заявки...")
                
                # Получение FIGI
                figi = get_figi_with_moex_store(self.current_signal['ticker'], token, self.logger)
                
                if not figi:
                    self.logger.error("FIGI не найден")
                    self.message_queue.put(('status', "Ошибка: FIGI не найден"))
                    return
                
                # Получение account_id
                from main import get_account_id
                account_id = get_account_id(token, self.logger)
                
                if not account_id:
                    self.logger.error("Account ID не получен")
                    self.message_queue.put(('status', "Ошибка: Account ID не получен"))
                    return
                
                # Выставление заявки
                success = place_order(self.current_signal, figi, token, account_id, self.logger)
                
                if success:
                    self.message_queue.put(('status', "✅ Заявка успешно выставлена"))
                    self.logger.info("Заявка успешно выставлена!")
                    messagebox.showinfo("Успех", "Заявка успешно выставлена!\nSL и TP установлены.")
                else:
                    self.message_queue.put(('status', "❌ Ошибка выставления заявки"))
                    
            except Exception as e:
                self.logger.error(f"Критическая ошибка: {e}")
                self.message_queue.put(('status', f"Ошибка: {str(e)}"))
                messagebox.showerror("Ошибка", f"Ошибка торговли:\n{str(e)}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _show_copy_menu(self, event):
        """Показ контекстного меню копирования"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Копировать", command=lambda: self.root.focus_get().event_generate("<<Copy>>"))
        menu.tk_popup(event.x_root, event.y_root)
    
    def _save_log(self):
        """Сохранение лога в файл"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_console.get("1.0", tk.END))
                messagebox.showinfo("Успех", f"Лог сохранён в:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить лог:\n{str(e)}")
    
    def _show_trade_settings(self):
        """Показ диалога настроек торговли"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Параметры торговли")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Настройки торговых параметров", font=self.fonts['title']).pack(pady=10)
        
        # Здесь можно добавить дополнительные настройки
        ttk.Label(dialog, text="Функционал в разработке...").pack(pady=20)
        
        ttk.Button(dialog, text="Закрыть", command=dialog.destroy).pack(pady=10)
    
    def _show_token_settings(self):
        """Показ диалога настройки токена"""
        dialog = tk.Toplevel(self.root)
        dialog.title("API Токен T-Invest")
        dialog.geometry("500x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="T-Invest API Token", font=self.fonts['title']).pack(pady=10)
        
        ttk.Label(dialog, text="Токен:").pack(anchor="w", padx=20)
        token_entry = ttk.Entry(dialog, width=60, show="*")
        token_entry.pack(padx=20, pady=5)
        
        def save_token():
            token = token_entry.get().strip()
            if token:
                os.environ['INVEST_TOKEN'] = token
                messagebox.showinfo("Успех", "Токен сохранён в переменную окружения (текущая сессия)")
                dialog.destroy()
        
        ttk.Button(dialog, text="Сохранить", command=save_token).pack(pady=10)
        
        ttk.Label(dialog, text="⚠️ Токен хранится только в текущей сессии", 
                 font=self.fonts['small'], foreground='orange').pack(pady=5)
    
    def _show_about(self):
        """Показ окна 'О программе'"""
        messagebox.showinfo(
            "О программе",
            "Lag-Llama Trading System v2.0\n\n"
            "Графический интерфейс для системы алгоритмической торговли\n"
            "на основе модели Lag-Llama и брокера T-Bank.\n\n"
            "© 2024 All rights reserved."
        )
    
    def _show_docs(self):
        """Показ документации"""
        docs_window = tk.Toplevel(self.root)
        docs_window.title("Документация")
        docs_window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(docs_window, wrap=tk.WORD, font=self.fonts['normal'])
        text.pack(fill="both", expand=True, padx=10, pady=10)
        
        doc_content = """
LAG-LLAMA TRADING SYSTEM - ДОКУМЕНТАЦИЯ

1. НАЧАЛО РАБОТЫ:
   - Введите тикер инструмента (например, SNGS, SBER, GAZP)
   - Нажмите "📊 Получить сигнал" для анализа
   
2. ПАРАМЕТРЫ СИГНАЛА:
   - Min Confidence: минимальная уверенность модели (0-100%)
   - Min R/R Ratio: минимальное соотношение риск/прибыль
   - Min Profit: минимальный прогнозируемый profit в %
   
3. ТОРГОВЛЯ:
   - После получения валидного сигнала кнопка "💰 Выставить заявку" станет активной
   - Убедитесь, что INVEST_TOKEN установлен
   - Нажмите кнопку для выставления заявки с SL/TP
   
4. ТРЕБОВАНИЯ:
   - Python 3.8+
   - Установленные зависимости: lag-llama, moexalgo, t_tech.invest
   - Активный брокерский счет T-Bank
   - API токен T-Invest
   
5. БЕЗОПАСНОСТЬ:
   - Не используйте токен на публичных компьютерах
   - Проверяйте параметры перед выставлением заявки
   - Используйте демо-счет для тестирования
        """
        
        text.insert("1.0", doc_content)
        text.config(state='disabled')
    
    def _on_closing(self):
        """Обработчик закрытия окна"""
        if messagebox.askokcancel("Выход", "Завершить работу приложения?"):
            self.root.destroy()


def main():
    """Точка входа приложения"""
    root = tk.Tk()
    
    # Установка иконки (если доступна)
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
    
    app = TradingGUI(root)
    
    # Обработчик закрытия
    root.protocol("WM_DELETE_WINDOW", app._on_closing)
    
    # Запуск главного цикла
    root.mainloop()


if __name__ == "__main__":
    main()
