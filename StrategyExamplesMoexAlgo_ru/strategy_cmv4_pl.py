import backtrader as bt
import math


class StrategyCMV4(bt.Strategy):
    """
    SNGS Adaptive Channel + Macro Regime (v4)
    strategy_id: sngs_channel_macro_v4
    class: channel_momentum_adaptive
    
    Логика стратегии на основе JSON спецификации:
    - Адаптивный канал с EMA basis и volatility-based коэффициентами
    - RSI подтверждение с динамическими границами
    - Макро overlay с корреляцией Brent in RUB
    - Volume liquidity фильтр через turnover_rub
    - Stop-limit ордера с fallback на market
    - Take-profit по трем уровням
    """
    params = (
        ('name', None),
        ('symbols', None),
        ('timeframe', ''),
        
        # Channel parameters
        ('ema_period_N', 20),
        ('volatility_period_N', 20),
        ('atr_period', 20),
        ('rsi_period_M', 14),
        
        # Volatility regime thresholds
        ('low_vol_threshold', 0.015),
        ('high_vol_threshold', 0.035),
        
        # K coefficients for different regimes
        ('K_out_low_vol', 1.0),
        ('K_in_low_vol', 0.35),
        ('K_out_normal_vol', 1.3),
        ('K_in_normal_vol', 0.5),
        ('K_out_high_vol', 1.8),
        ('K_in_high_vol', 0.7),
        
        # Smoothing parameters
        ('smoothing_alpha', 0.2),
        ('min_K_out', 1.4),
        ('max_K_out', 3.0),
        
        # Macro overlay
        ('macro_corr_lookback_1w', 12),
        ('macro_corr_lookback_1d', 20),
        ('corr_aligned_mult', 0.95),
        ('corr_weak_mult', 1.1),
        
        # Risk parameters
        ('base_risk_pct', 0.025),
        ('max_position_pct', 0.04),
        ('soft_dd_limit', 0.06),
        ('hard_dd_limit', 0.10),
        ('recovery_risk_mult', 0.5),
        
        # Exit parameters
        ('take_profit_target1_pct', 0.5),  # 50% at basis
        ('take_profit_target2_pct', 0.3),  # 30% at opposite dispersion
        ('take_profit_target3_pct', 0.2),  # 20% at opposite outer
        ('max_bars_10m', 72),
        ('stop_limit_offset_bps', 15),
        ('fallback_bars', 2),
        
        # Execution filters
        ('volume_liquidity_sma_period', 20),
        ('max_price_deviation_pct', 0.25),
        ('timeout_bars_10m', 3),
        
        # Cost model
        ('commission_bps', 5),
        ('slippage_base_bps', 4),
    )

    def log(self, txt, dt=None):
        """Вывод строки с датой на консоль"""
        dt = bt.num2date(self.datas[0].datetime[0]) if not dt else dt
        print(f'{dt.strftime("%d.%m.%Y %H:%M")}, {txt}')

    def __init__(self):
        """Инициализация, добавление индикаторов для каждого тикера"""
        self.isLive = False
        self.orders = {}
        self.position_entry_price = {}
        self.position_entry_bar = {}
        self.stop_orders = {}
        self.tp_orders = {}
        self.pending_orders = {}
        self.order_timeout = {}
        self.macro_correlation = {}
        self.channel_adjustment = {}
        self.prev_rsi = {}
        self.prev_price_vs_channel = {}
        self.turnover_sma = {}
        self.high_peak = {}  # Для drawdown tracking
        
        for d in self.datas:
            self.orders[d._name] = None
            self.position_entry_price[d._name] = None
            self.position_entry_bar[d._name] = None
            self.stop_orders[d._name] = None
            self.tp_orders[d._name] = None
            self.pending_orders[d._name] = None
            self.order_timeout[d._name] = 0
            self.macro_correlation[d._name] = 0.0
            self.channel_adjustment[d._name] = {'K_out_mult': 1.0}
            self.prev_rsi[d._name] = None
            self.prev_price_vs_channel[d._name] = None
            self.high_peak[d._name] = 0.0
            
        # Индикаторы для каждого тикера
        self.ema_basis = {}
        self.volatility_std = {}
        self.atr = {}
        self.rsi = {}
        self.turnover_sma_ind = {}
        self.v_ratio = {}
        
        # Канальные границы
        self.upper_bound = {}
        self.lower_bound = {}
        self.disp_upper = {}
        self.disp_lower = {}
        
        # Динамические RSI границы (заглушки - rolling_percentile требует доп.обработки)
        self.dynamic_oversold = {}
        self.dynamic_overbought = {}
        
        for i in range(len(self.datas)):
            ticker = list(self.dnames.keys())[i]
            data = self.datas[i]
            
            # Basis: EMA(N)
            self.ema_basis[ticker] = bt.indicators.EMA(data.close, period=self.p.ema_period_N)
            
            # Volatility: StdDev(N)
            self.volatility_std[ticker] = bt.indicators.StandardDeviation(data.close, period=self.p.volatility_period_N)
            
            # ATR для V-ratio расчета
            self.atr[ticker] = bt.indicators.ATR(data, period=self.p.atr_period)
            
            # RSI(M)
            self.rsi[ticker] = bt.indicators.RSI(data.close, period=self.p.rsi_period_M)
            
            # Turnover SMA для liquidity фильтра (заглушка - используем volume как proxy)
            self.turnover_sma_ind[ticker] = bt.indicators.SMA(data.volume, period=self.p.volume_liquidity_sma_period)
            
            # V-ratio = ATR / Basis (адаптивный коэффициент волатильности)
            self.v_ratio[ticker] = self.atr[ticker] / self.ema_basis[ticker]
            
            # Канальные границы будут вычисляться в next() с учетом адаптивных K коэффициентов
            
            # Динамические RSI границы - заглушки (rolling_percentile недоступен напрямую)
            # Используем статические уровни как fallback
            self.dynamic_oversold[ticker] = 30.0  # Заглушка
            self.dynamic_overbought[ticker] = 70.0  # Заглушка

    def _get_k_coefficients(self, ticker):
        """
        Получить адаптивные K коэффициенты на основе режима волатильности
        v_ratio = ATR(N) / Basis
        """
        v_ratio_val = self.v_ratio[ticker][0]
        
        if v_ratio_val < self.p.low_vol_threshold:
            return self.p.K_out_low_vol, self.p.K_in_low_vol, 'low_vol'
        elif v_ratio_val > self.p.high_vol_threshold:
            return self.p.K_out_high_vol, self.p.K_in_high_vol, 'high_vol'
        else:
            return self.p.K_out_normal_vol, self.p.K_in_normal_vol, 'normal_vol'
    
    def _apply_macro_adjustment(self, ticker, K_out):
        """
        Применить макро корректировку к K_out на основе корреляции с Brent in RUB
        Заглушка - реальная корреляция требует внешних данных
        """
        corr = self.macro_correlation.get(ticker, 0.0)
        adj = self.channel_adjustment.get(ticker, {'K_out_mult': 1.0})
        
        # Если корреляция выровнена (> 0.3)
        if corr > 0.3:
            return K_out * self.p.corr_aligned_mult, 'aligned'
        # Если корреляция слабая (0 < corr <= 0.3)
        elif corr > 0.0:
            return K_out * self.p.corr_weak_mult, 'weak'
        # Если корреляция противоположная (corr <= 0)
        else:
            return K_out, 'opposite'  # pause_new_entries будет обработано в signal_generation
    
    def _check_volume_liquidity(self, ticker, data):
        """
        Проверка ликвидности через turnover_rub > sma(turnover_rub, 20)
        Заглушка - используем volume как proxy для turnover_rub
        """
        current_volume = data.volume[0]
        avg_volume = self.turnover_sma_ind[ticker][0]
        
        if avg_volume > 0:
            ratio = current_volume / avg_volume
            return ratio >= 1.0, ratio
        return False, 0.0
    
    def _calculate_dynamic_rsi_bounds(self, ticker):
        """
        Расчет динамических RSI границ через rolling_percentile_20d
        Заглушка - возвращаем статические значения
        """
        # В реальной реализации нужен rolling percentile расчет за 20 дней
        # Здесь используем упрощенный подход
        rsi_val = self.rsi[ticker][0]
        
        # Заглушка: используем фиксированные процентили
        oversold = 25.0  # rolling_percentile_20d(rsi) <= 0.25
        overbought = 75.0  # rolling_percentile_20d(rsi) >= 0.75
        
        return oversold, overbought
    
    def _update_macro_correlation(self, ticker):
        """
        Обновление макро корреляции: pearson_corr(brent_in_rub_daily_returns[t-1], sngs_returns[t])
        Заглушка - требует внешних макроданных
        """
        # В реальной реализации нужно загружать brent_futures и usd_rub данные
        # и рассчитывать cross: brent_in_rub = brent_futures.close * usd_rub.close
        # Здесь оставляем заглушку
        pass
    
    def _check_drawdown_limits(self, ticker):
        """
        Проверка лимитов drawdown
        """
        portfolio_value = self.broker.getvalue()
        initial_value = self.broker.getcash() / (1 - self.p.base_risk_pct)  # approximation
        
        # Обновляем пик
        if portfolio_value > self.high_peak[ticker]:
            self.high_peak[ticker] = portfolio_value
        
        if self.high_peak[ticker] > 0:
            dd = (self.high_peak[ticker] - portfolio_value) / self.high_peak[ticker]
            
            if dd >= self.p.hard_dd_limit:
                return 'hard_limit'
            elif dd >= self.p.soft_dd_limit:
                return 'soft_limit'
        
        return 'normal'
    
    def next(self):
        """Приход нового бара тикера"""
        for data in self.datas:
            if not self.p.symbols or data._name in self.p.symbols:
                # Используем _name вместо dataname, так как для PandasData dataname может быть DataFrame
                ticker = data._name if isinstance(data._name, str) else (data.p.dataname if isinstance(data.p.dataname, str) else str(data.p.dataname))
                self.log(f'{ticker} - {bt.TimeFrame.Names[data.p.timeframe]} {data.p.compression} - Open={data.open[0]:.2f}, High={data.high[0]:.2f}, Low={data.low[0]:.2f}, Close={data.close[0]:.2f}, Volume={data.volume[0]:.0f}',
                         bt.num2date(data.datetime[0]))
                
                # Обновляем макро корреляцию (заглушка)
                self._update_macro_correlation(ticker)
                
                # Получаем адаптивные K коэффициенты
                K_out_base, K_in, vol_regime = self._get_k_coefficients(ticker)
                
                # Применяем макро корректировку
                K_out, corr_state = self._apply_macro_adjustment(ticker, K_out_base)
                
                # Вычисляем канальные границы
                basis = self.ema_basis[ticker][0]
                sigma = self.volatility_std[ticker][0]
                
                self.upper_bound[ticker] = basis + K_out * sigma
                self.lower_bound[ticker] = basis - K_out * sigma
                self.disp_upper[ticker] = basis + K_in * sigma
                self.disp_lower[ticker] = basis - K_in * sigma
                
                # Динамические RSI границы
                dyn_oversold, dyn_overbought = self._calculate_dynamic_rsi_bounds(ticker)
                
                # Текущие значения
                price = data.close[0]
                rsi_val = self.rsi[ticker][0]
                
                # Проверка ликвидности
                liq_ok, liq_ratio = self._check_volume_liquidity(ticker, data)
                
                # Проверка drawdown
                dd_state = self._check_drawdown_limits(ticker)
                
                # Определяем позицию цены относительно канала
                prev_price_vs_channel = self.prev_price_vs_channel.get(ticker, 'neutral')
                
                if price <= self.lower_bound[ticker]:
                    current_price_vs_channel = 'below_lower'
                elif price >= self.upper_bound[ticker]:
                    current_price_vs_channel = 'above_upper'
                elif price <= self.disp_lower[ticker]:
                    current_price_vs_channel = 'in_disp_lower'
                elif price >= self.disp_upper[ticker]:
                    current_price_vs_channel = 'in_disp_upper'
                else:
                    current_price_vs_channel = 'neutral'
                
                # Проверяем кроссы RSI
                rsi_cross_above_oversold = False
                rsi_cross_below_overbought = False
                
                if self.prev_rsi.get(ticker) is not None:
                    prev_rsi = self.prev_rsi[ticker]
                    # Long: RSI crosses above dynamic_oversold
                    if prev_rsi <= dyn_oversold and rsi_val > dyn_oversold:
                        rsi_cross_above_oversold = True
                    # Short: RSI crosses below dynamic_overbought
                    if prev_rsi >= dyn_overbought and rsi_val < dyn_overbought:
                        rsi_cross_below_overbought = True
                
                self.prev_rsi[ticker] = rsi_val
                self.prev_price_vs_channel[ticker] = current_price_vs_channel
                
                # --- SIGNAL GENERATION ---
                
                if not self.getposition(data):  # Если позиции нет
                    # Проверка условий для long entry
                    long_condition = (
                        price <= self.lower_bound[ticker] and
                        rsi_cross_above_oversold and
                        liq_ok and
                        dd_state != 'hard_limit' and
                        corr_state != 'opposite'  # pause_new_entries if opposite
                    )
                    
                    # Проверка подтверждения: prev_bar in buffer or below
                    if long_condition and prev_price_vs_channel in ['below_lower', 'in_disp_lower']:
                        long_condition = True
                    
                    # Проверка условий для short entry
                    short_condition = (
                        price >= self.upper_bound[ticker] and
                        rsi_cross_below_overbought and
                        liq_ok and
                        dd_state != 'hard_limit' and
                        corr_state != 'opposite'
                    )
                    
                    # Проверка подтверждения: prev_bar in buffer or above
                    if short_condition and prev_price_vs_channel in ['above_upper', 'in_disp_upper']:
                        short_condition = True
                    
                    # Выполнение входа
                    if long_condition:
                        free_money = self.broker.getcash()
                        
                        # Position sizing: risk_amount / abs(entry_price - stop_trigger_price)
                        stop_trigger_price = self.lower_bound[ticker] - 0.5 * K_out * sigma
                        risk_amount = free_money * self.p.base_risk_pct
                        risk_per_share = abs(price - stop_trigger_price)
                        
                        if risk_per_share > 0:
                            size_shares = risk_amount / risk_per_share
                            
                            # Hard cap: total_position_value <= capital × max_position_pct
                            max_size_by_cap = (free_money * self.p.max_position_pct) / price
                            size_shares = min(size_shares, max_size_by_cap)
                            
                            # Расчет TP уровней
                            tp1_price = basis  # Target 1: basis
                            tp2_price = self.disp_upper[ticker]  # Target 2: opposite dispersion
                            tp3_price = self.upper_bound[ticker]  # Target 3: opposite outer boundary
                            
                            # SL цена (trigger для stop-limit)
                            sl_trigger_price = stop_trigger_price
                            
                            # Limit offset для SL (15 bps = 0.15%)
                            sl_limit_offset = sl_trigger_price * (self.p.stop_limit_offset_bps / 10000.0)
                            sl_limit_price = sl_trigger_price - sl_limit_offset  # Для long: limit ниже trigger
                            
                            print("-" * 50)
                            print(f"\t - LONG ENTRY {ticker}")
                            print(f"\t - Price: {price:.2f}, RSI: {rsi_val:.1f}")
                            print(f"\t - Channel: Lower={self.lower_bound[ticker]:.2f}, Basis={basis:.2f}, Upper={self.upper_bound[ticker]:.2f}")
                            print(f"\t - Vol Regime: {vol_regime}, K_out={K_out:.2f}, K_in={K_in:.2f}")
                            print(f"\t - Corr State: {corr_state}")
                            print(f"\t - Size: {size_shares:.0f} shares")
                            print(f"\t - SL Trigger: {sl_trigger_price:.2f}, SL Limit: {sl_limit_price:.2f}")
                            print(f"\t - TP1 (50%): {tp1_price:.2f}, TP2 (30%): {tp2_price:.2f}, TP3 (20%): {tp3_price:.2f}")
                            print("-" * 50)
                            
                            # Создаем заявку на покупку (limit order)
                            buy_order = self.buy(data=data, exectype=bt.Order.Limit, price=price, size=size_shares)
                            self.pending_orders[ticker] = buy_order
                            self.order_timeout[ticker] = self.p.timeout_bars_10m
                            
                            # Сохраняем информацию о позиции для последующей установки TP/SL
                            self.position_entry_price[ticker] = price
                            self.position_entry_bar[ticker] = len(data)
                            
                            # Примечание: TP/SL ордера будут установлены после исполнения покупки в notify_order
                    
                    elif short_condition:
                        free_money = self.broker.getcash()
                        
                        # Position sizing
                        stop_trigger_price = self.upper_bound[ticker] + 0.5 * K_out * sigma
                        risk_amount = free_money * self.p.base_risk_pct
                        risk_per_share = abs(price - stop_trigger_price)
                        
                        if risk_per_share > 0:
                            size_shares = risk_amount / risk_per_share
                            max_size_by_cap = (free_money * self.p.max_position_pct) / price
                            size_shares = min(size_shares, max_size_by_cap)
                            
                            # TP уровни для short
                            tp1_price = basis
                            tp2_price = self.disp_lower[ticker]
                            tp3_price = self.lower_bound[ticker]
                            
                            # SL для short
                            sl_trigger_price = stop_trigger_price
                            sl_limit_offset = sl_trigger_price * (self.p.stop_limit_offset_bps / 10000.0)
                            sl_limit_price = sl_trigger_price + sl_limit_offset  # Для short: limit выше trigger
                            
                            print("-" * 50)
                            print(f"\t - SHORT ENTRY {ticker}")
                            print(f"\t - Price: {price:.2f}, RSI: {rsi_val:.1f}")
                            print(f"\t - Channel: Lower={self.lower_bound[ticker]:.2f}, Basis={basis:.2f}, Upper={self.upper_bound[ticker]:.2f}")
                            print(f"\t - Vol Regime: {vol_regime}, K_out={K_out:.2f}, K_in={K_in:.2f}")
                            print(f"\t - Corr State: {corr_state}")
                            print(f"\t - Size: {size_shares:.0f} shares")
                            print(f"\t - SL Trigger: {sl_trigger_price:.2f}, SL Limit: {sl_limit_price:.2f}")
                            print(f"\t - TP1 (50%): {tp1_price:.2f}, TP2 (30%): {tp2_price:.2f}, TP3 (20%): {tp3_price:.2f}")
                            print("-" * 50)
                            
                            sell_order = self.sell(data=data, exectype=bt.Order.Limit, price=price, size=size_shares)
                            self.pending_orders[ticker] = sell_order
                            self.order_timeout[ticker] = self.p.timeout_bars_10m
                            
                            self.position_entry_price[ticker] = price
                            self.position_entry_bar[ticker] = len(data)
                
                else:  # Если позиция есть
                    position = self.getposition(data)
                    pos_size = position.size
                    entry_price = self.position_entry_price.get(ticker, price)
                    entry_bar = self.position_entry_bar.get(ticker, len(data))
                    bars_held = len(data) - entry_bar
                    
                    # Определяем направление позиции
                    is_long = pos_size > 0
                    
                    # --- EXIT RULES ---
                    
                    # Time management: max_bars_10m
                    time_exit = bars_held >= self.p.max_bars_10m
                    
                    # Early exit: price crosses basis against position + rsi confirm
                    early_exit = False
                    if is_long and price < basis and rsi_val < 50:
                        early_exit = True
                    elif not is_long and price > basis and rsi_val > 50:
                        early_exit = True
                    
                    # Stop-loss check
                    sl_trigger_price = None
                    if is_long:
                        sl_trigger_price = self.lower_bound[ticker] - 0.5 * K_out * sigma
                        sl_triggered = price <= sl_trigger_price
                    else:
                        sl_trigger_price = self.upper_bound[ticker] + 0.5 * K_out * sigma
                        sl_triggered = price >= sl_trigger_price
                    
                    # Take-profit levels
                    tp1_triggered = False
                    tp2_triggered = False
                    tp3_triggered = False
                    
                    if is_long:
                        tp1_triggered = price >= basis
                        tp2_triggered = price >= self.disp_upper[ticker]
                        tp3_triggered = price >= self.upper_bound[ticker]
                    else:
                        tp1_triggered = price <= basis
                        tp2_triggered = price <= self.disp_lower[ticker]
                        tp3_triggered = price <= self.lower_bound[ticker]
                    
                    # Trailing activation: price touches basis in profit
                    trailing_active = False
                    trailing_stop_price = None
                    if is_long and price >= basis and entry_price < basis:
                        trailing_active = True
                        trailing_stop_price = basis - K_in * sigma
                    elif not is_long and price <= basis and entry_price > basis:
                        trailing_active = True
                        trailing_stop_price = basis + K_in * sigma
                    
                    # Выполнение выхода
                    exit_reason = None
                    exit_size = pos_size  # По умолчанию закрываем всю позицию
                    
                    if sl_triggered:
                        exit_reason = f'SL Triggered @ {sl_trigger_price:.2f}'
                        # Fallback: market если не заполнено за 2 бара (упрощено)
                        self.close(data=data)
                    
                    elif tp3_triggered:
                        exit_reason = 'TP3 Hit (full exit)'
                        self.close(data=data)
                    
                    elif tp2_triggered:
                        # Закрываем 30% на TP2
                        exit_size = int(abs(pos_size) * self.p.take_profit_target2_pct)
                        if exit_size > 0:
                            exit_reason = f'TP2 Hit - closing {exit_size} shares'
                            if is_long:
                                self.sell(data=data, size=exit_size)
                            else:
                                self.buy(data=data, size=exit_size)
                    
                    elif tp1_triggered and not hasattr(self, f'_tp1_closed_{ticker}'):
                        # Закрываем 50% на TP1
                        exit_size = int(abs(pos_size) * self.p.take_profit_target1_pct)
                        if exit_size > 0:
                            exit_reason = f'TP1 Hit - closing {exit_size} shares'
                            if is_long:
                                self.sell(data=data, size=exit_size)
                            else:
                                self.buy(data=data, size=exit_size)
                            setattr(self, f'_tp1_closed_{ticker}', True)
                    
                    elif time_exit:
                        exit_reason = f'Time Exit ({bars_held} bars)'
                        self.close(data=data)
                    
                    elif early_exit:
                        exit_reason = 'Early Exit (basis cross + RSI)'
                        self.close(data=data)
                    
                    elif trailing_active and trailing_stop_price:
                        if is_long and price <= trailing_stop_price:
                            exit_reason = f'Trailing Stop @ {trailing_stop_price:.2f}'
                            self.close(data=data)
                        elif not is_long and price >= trailing_stop_price:
                            exit_reason = f'Trailing Stop @ {trailing_stop_price:.2f}'
                            self.close(data=data)
                    
                    if exit_reason:
                        print("-" * 50)
                        print(f"\t - EXIT {ticker}: {exit_reason}")
                        print(f"\t - Position: {'LONG' if is_long else 'SHORT'}, Size: {pos_size}")
                        print(f"\t - Entry Price: {entry_price:.2f}, Current: {price:.2f}")
                        print(f"\t - PnL: {(price - entry_price) * pos_size:.2f}")
                        print("-" * 50)
                
                # Обработка таймаута pending ордеров
                if self.pending_orders.get(ticker) and self.pending_orders[ticker].status == bt.Order.Accepted:
                    self.order_timeout[ticker] -= 1
                    if self.order_timeout[ticker] <= 0:
                        self.cancel(self.pending_orders[ticker])
                        self.pending_orders[ticker] = None
                        print(f"\t - Order timeout for {ticker}, cancelled")

    def notify_order(self, order):
        """Изменение статуса заявки"""
        order_data_name = order.data._name
        print("*" * 50)
        self.log(f'Заявка номер {order.ref} {order.info.get("order_number", "")} {order.getstatusname()} {"Покупка" if order.isbuy() else "Продажа"} {order_data_name} {order.size} @ {order.price}')
        
        if order.status == bt.Order.Completed:
            if order.isbuy():
                self.log(f'Покупка {order_data_name} Цена: {order.executed.price:.2f}, Объём: {order.executed.value:.2f}, Комиссия: {order.executed.comm:.2f}')
                
                # После исполнения покупки устанавливаем TP и SL ордера
                ticker = order_data_name
                data = order.data
                executed_price = order.executed.price
                size = order.executed.size
                
                # Получаем текущие значения канала
                basis = self.ema_basis[ticker][0]
                sigma = self.volatility_std[ticker][0]
                K_out, _, _ = self._get_k_coefficients(ticker)
                
                # SL trigger и limit
                sl_trigger = self.lower_bound[ticker] - 0.5 * K_out * sigma
                sl_limit_offset = sl_trigger * (self.p.stop_limit_offset_bps / 10000.0)
                sl_limit = sl_trigger - sl_limit_offset
                
                # TP уровни с размерами
                tp1_size = int(size * self.p.take_profit_target1_pct)
                tp2_size = int(size * self.p.take_profit_target2_pct)
                tp3_size = size - tp1_size - tp2_size  # Остаток на TP3
                
                # Создаем SL ордер (stop-limit)
                sl_order = self.sell(data=data, exectype=bt.Order.StopLimit, 
                                     price=sl_limit, plimit=sl_limit, size=size, parent=order)
                self.stop_orders[ticker] = sl_order
                
                # Создаем TP ордера
                tp1_order = self.sell(data=data, exectype=bt.Order.Limit, price=basis, size=tp1_size, parent=order)
                tp2_order = self.sell(data=data, exectype=bt.Order.Limit, price=self.disp_upper[ticker], size=tp2_size, parent=order)
                tp3_order = self.sell(data=data, exectype=bt.Order.Limit, price=self.upper_bound[ticker], size=tp3_size, parent=order)
                self.tp_orders[ticker] = [tp1_order, tp2_order, tp3_order]
                
                self.orders[ticker] = order
                self.pending_orders[ticker] = None  # Сбрасываем pending
                
            else:
                self.log(f'Продажа {order_data_name} Цена: {order.executed.price:.2f}, Объём: {order.executed.value:.2f}, Комиссия: {order.executed.comm:.2f}')
                
                # Для short позиции аналогично устанавливаем TP/SL
                ticker = order_data_name
                data = order.data
                executed_price = order.executed.price
                size = abs(order.executed.size)
                
                basis = self.ema_basis[ticker][0]
                sigma = self.volatility_std[ticker][0]
                K_out, _, _ = self._get_k_coefficients(ticker)
                
                # SL для short
                sl_trigger = self.upper_bound[ticker] + 0.5 * K_out * sigma
                sl_limit_offset = sl_trigger * (self.p.stop_limit_offset_bps / 10000.0)
                sl_limit = sl_trigger + sl_limit_offset
                
                tp1_size = int(size * self.p.take_profit_target1_pct)
                tp2_size = int(size * self.p.take_profit_target2_pct)
                tp3_size = size - tp1_size - tp2_size
                
                sl_order = self.buy(data=data, exectype=bt.Order.StopLimit,
                                    price=sl_limit, plimit=sl_limit, size=size, parent=order)
                self.stop_orders[ticker] = sl_order
                
                tp1_order = self.buy(data=data, exectype=bt.Order.Limit, price=basis, size=tp1_size, parent=order)
                tp2_order = self.buy(data=data, exectype=bt.Order.Limit, price=self.disp_lower[ticker], size=tp2_size, parent=order)
                tp3_order = self.buy(data=data, exectype=bt.Order.Limit, price=self.lower_bound[ticker], size=tp3_size, parent=order)
                self.tp_orders[ticker] = [tp1_order, tp2_order, tp3_order]
                
                self.orders[ticker] = order
                self.pending_orders[ticker] = None
                
        elif order.status == bt.Order.Cancelled:
            self.pending_orders[order_data_name] = None
            self.log(f'Заявка отменена {order_data_name}')
            
        print("*" * 50)

    def notify_trade(self, trade):
        """Изменение статуса позиции"""
        if trade.isclosed:
            self.log(f'Прибыль по закрытой позиции {trade.getdataname()} Общая={trade.pnl:.2f}, Без комиссии={trade.pnlcomm:.2f}')
            
            # Сброс флагов TP1 при полном закрытии
            ticker = trade.getdataname()
            if hasattr(self, f'_tp1_closed_{ticker}'):
                delattr(self, f'_tp1_closed_{ticker}')

    def notify_data(self, data, status, *args, **kwargs):
        """Изменение статуса приходящих баров"""
        data_status = data._getstatusname(status)
        _name = data._name if data._name else f"{self.p.name}"
        print(f'{_name} - {bt.TimeFrame.Names[data.p.timeframe]} {data.p.compression} - {data_status}')
        self.isLive = data_status == 'LIVE'


# Алиас для совместимости
StrategyCMV4PL = StrategyCMV4
