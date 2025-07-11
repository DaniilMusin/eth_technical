#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 РЕАЛЬНОЕ WALK FORWARD ТЕСТИРОВАНИЕ
Упрощенная версия для запуска с настоящими данными ETH
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimpleWalkForwardTester:
    """Упрощенная система walk forward тестирования"""
    
    def __init__(self, data_path, train_months=6, test_months=1, initial_balance=10000):
        self.data_path = data_path
        self.train_months = train_months
        self.test_months = test_months
        self.initial_balance = initial_balance
        self.results = []
        
    def load_data(self):
        """Загружаем и подготавливаем данные"""
        print("📊 Загружаем данные...")
        
        df = pd.read_csv(self.data_path)
        
        # Стандартизируем названия колонок
        rename_map = {
            "open_time": "timestamp",
            "open": "open",
            "high": "high", 
            "low": "low",
            "close": "close",
            "volume": "volume"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Конвертируем время
        if df["timestamp"].dtype.kind in "iu":
            if df["timestamp"].max() < 1e12:
                # Likely seconds
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        df.dropna(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True)
        
        # Конвертируем цены в числовой формат
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        
        print(f"✅ Данные загружены: {len(df):,} записей")
        print(f"📅 Период: {df.index[0]} - {df.index[-1]}")
        
        self.data = df
        return df
    
    def calculate_simple_indicators(self, data):
        """Рассчитываем простые технические индикаторы"""
        df = data.copy()
        
        # Скользящие средние
        df['sma_fast'] = df['close'].rolling(window=8).mean()
        df['sma_slow'] = df['close'].rolling(window=25).mean()
        
        # EMA
        df['ema_fast'] = df['close'].ewm(span=8).mean()
        df['ema_slow'] = df['close'].ewm(span=25).mean()
        
        # RSI (упрощенный)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volatility (ATR упрощенный)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Объем
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def generate_signals(self, data):
        """Генерируем торговые сигналы"""
        df = data.copy()
        
        # Условия для лонга
        long_conditions = (
            (df['ema_fast'] > df['ema_slow']) &  # Восходящий тренд
            (df['rsi'] < 70) &  # Не перекуплен
            (df['close'] > df['bb_lower']) &  # Выше нижней полосы
            (df['volume_ratio'] > 1.1)  # Повышенный объем
        )
        
        # Условия для шорта  
        short_conditions = (
            (df['ema_fast'] < df['ema_slow']) &  # Нисходящий тренд
            (df['rsi'] > 30) &  # Не перепродан
            (df['close'] < df['bb_upper']) &  # Ниже верхней полосы
            (df['volume_ratio'] > 1.1)  # Повышенный объем
        )
        
        df['long_signal'] = long_conditions
        df['short_signal'] = short_conditions
        
        return df
    
    def run_strategy_backtest(self, data):
        """Запускаем бэктест стратегии на данных"""
        df = self.calculate_simple_indicators(data)
        df = self.generate_signals(df)
        
        # Убираем NaN
        df = df.dropna()
        
        if len(df) < 100:
            return self.get_empty_result()
        
        # Торговая симуляция
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        max_equity = balance
        max_drawdown = 0
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            equity = balance + (position * (current_price - entry_price) if position != 0 else 0)
            equity_curve.append(equity)
            
            # Обновляем максимальную просадку
            if equity > max_equity:
                max_equity = equity
            current_drawdown = (max_equity - equity) / max_equity * 100
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            
            # Закрытие позиций по стоп-лоссу/тейк-профиту
            if position != 0:
                atr = df['atr'].iloc[i]
                if position > 0:  # Лонг
                    stop_loss = entry_price - (atr * 2.0)
                    take_profit = entry_price + (atr * 5.0)
                    
                    if current_price <= stop_loss or current_price >= take_profit:
                        pnl = (current_price - entry_price) / entry_price * 100
                        balance = balance + (position * (current_price - entry_price))
                        trades.append({'type': 'close_long', 'pnl': pnl, 'price': current_price})
                        position = 0
                        entry_price = 0
                        
                elif position < 0:  # Шорт
                    stop_loss = entry_price + (atr * 2.0)
                    take_profit = entry_price - (atr * 5.0)
                    
                    if current_price >= stop_loss or current_price <= take_profit:
                        pnl = (entry_price - current_price) / entry_price * 100
                        balance = balance + (abs(position) * (entry_price - current_price))
                        trades.append({'type': 'close_short', 'pnl': pnl, 'price': current_price})
                        position = 0
                        entry_price = 0
            
            # Новые сигналы
            if position == 0:  # Нет позиции
                risk_amount = balance * 0.02  # 2% риска
                
                if df['long_signal'].iloc[i]:
                    atr = df['atr'].iloc[i]
                    stop_distance = atr * 2.0
                    position_size = risk_amount / stop_distance
                    position = position_size
                    entry_price = current_price
                    trades.append({'type': 'open_long', 'price': current_price})
                    
                elif df['short_signal'].iloc[i]:
                    atr = df['atr'].iloc[i]
                    stop_distance = atr * 2.0
                    position_size = risk_amount / stop_distance
                    position = -position_size
                    entry_price = current_price
                    trades.append({'type': 'open_short', 'price': current_price})
        
        # Закрываем финальную позицию
        if position != 0:
            final_price = df['close'].iloc[-1]
            if position > 0:
                pnl = (final_price - entry_price) / entry_price * 100
                balance = balance + (position * (final_price - entry_price))
            else:
                pnl = (entry_price - final_price) / entry_price * 100  
                balance = balance + (abs(position) * (entry_price - final_price))
            trades.append({'type': 'close_final', 'pnl': pnl, 'price': final_price})
        
        # Рассчитываем метрики
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        pnl_trades = [t for t in trades if 'pnl' in t]
        if pnl_trades:
            winning_trades = [t for t in pnl_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(pnl_trades) * 100
            if len(winning_trades) > 0 and len(pnl_trades) > len(winning_trades):
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                avg_loss = np.mean([t['pnl'] for t in pnl_trades if t['pnl'] <= 0])
                if avg_loss == 0:
                    profit_factor = 1.0 if total_return > 0 else 0.5
                else:
                    profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * (len(pnl_trades) - len(winning_trades)))
            else:
                profit_factor = 1.0 if total_return > 0 else 0.5
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(pnl_trades),
            'profit_factor': profit_factor,
            'final_balance': balance
        }
    
    def get_empty_result(self):
        """Возвращает пустой результат для периодов с недостаточными данными"""
        return {
            'total_return': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0,
            'profit_factor': 0,
            'final_balance': self.initial_balance
        }
    
    def create_periods(self):
        """Создаем периоды для walk forward тестирования"""
        periods = []
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        
        current_date = start_date
        period_id = 1
        
        while True:
            # Период обучения (пропускаем, используем фиксированные параметры)
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_months)
            
            # Период тестирования
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            if test_end > end_date:
                break
            
            # Проверяем наличие данных
            test_data = self.data[test_start:test_end]
            
            if len(test_data) < 100:
                current_date = current_date + pd.DateOffset(months=self.test_months)
                period_id += 1
                continue
            
            periods.append({
                'id': period_id,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'test_records': len(test_data)
            })
            
            current_date = current_date + pd.DateOffset(months=self.test_months)
            period_id += 1
            
            # Ограничиваем количество периодов для демонстрации
            if len(periods) >= 20:
                break
        
        return periods
    
    def run_walk_forward_test(self):
        """Запуск walk forward тестирования"""
        print("\n🚀 ЗАПУСК РЕАЛЬНОГО WALK FORWARD ТЕСТИРОВАНИЯ")
        print("="*60)
        
        # Загружаем данные
        self.load_data()
        
        # Создаем периоды
        periods = self.create_periods()
        print(f"🔄 Создано {len(periods)} периодов для тестирования")
        
        print(f"\n📅 РЕЗУЛЬТАТЫ ПО ПЕРИОДАМ:")
        print("-" * 80)
        print(f"{'Period':<6} {'Test Start':<12} {'Test End':<12} {'Return %':<10} {'Drawdown %':<12} {'Win Rate %':<12} {'Trades':<7}")
        print("-" * 80)
        
        for period in periods:
            # Извлекаем тестовые данные
            test_data = self.data[period['test_start']:period['test_end']]
            
            # Запускаем стратегию (без оптимизации для упрощения)
            result = self.run_strategy_backtest(test_data)
            
            # Сохраняем результат
            period_result = {
                'period_id': period['id'],
                'test_start': period['test_start'].strftime('%Y-%m-%d'),
                'test_end': period['test_end'].strftime('%Y-%m-%d'),
                'test_records': period['test_records'],
                **result
            }
            
            self.results.append(period_result)
            
            # Выводим результат
            print(f"{period['id']:<6} {period['test_start'].strftime('%Y-%m-%d'):<12} {period['test_end'].strftime('%Y-%m-%d'):<12} "
                  f"{result['total_return']:>+7.2f} {result['max_drawdown']:>10.1f} {result['win_rate']:>10.1f} "
                  f"{result['total_trades']:>6}")
        
        # Анализируем общие результаты
        self.analyze_results()
        
        return self.results
    
    def analyze_results(self):
        """Анализируем общие результаты"""
        print(f"\n" + "="*60)
        print("📊 ОБЩИЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        if not self.results:
            print("❌ Нет результатов для анализа")
            return
        
        # Собираем метрики
        returns = [r['total_return'] for r in self.results]
        drawdowns = [r['max_drawdown'] for r in self.results]
        win_rates = [r['win_rate'] for r in self.results]
        trades = [r['total_trades'] for r in self.results]
        
        # Основная статистика
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        avg_drawdown = np.mean(drawdowns)
        max_drawdown = np.max(drawdowns)
        
        avg_win_rate = np.mean(win_rates)
        total_trades = np.sum(trades)
        
        profitable_periods = len([r for r in returns if r > 0])
        profitable_pct = profitable_periods / len(returns) * 100
        
        # Выводим статистику
        print(f"📊 Протестировано периодов: {len(self.results)}")
        print(f"📈 Средняя доходность за период: {avg_return:.2f}%")
        print(f"📈 Волатильность доходности: {std_return:.2f}%")
        print(f"📈 Диапазон доходности: {min_return:.2f}% - {max_return:.2f}%")
        print(f"📉 Средняя макс. просадка: {avg_drawdown:.2f}%")
        print(f"📉 Максимальная просадка: {max_drawdown:.2f}%")
        print(f"🎯 Средний Win Rate: {avg_win_rate:.1f}%")
        print(f"🔄 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных периодов: {profitable_periods} ({profitable_pct:.1f}%)")
        
        # Месячная доходность (периоды по 1 месяцу)
        monthly_return = avg_return
        annual_return = monthly_return * 12
        
        print(f"\n📅 ПРОЕКЦИЯ НА ГОД:")
        print(f"📈 Ожидаемая годовая доходность: {annual_return:.1f}%")
        
        # Стабильность
        if avg_return != 0:
            consistency_score = profitable_pct * (1 - std_return / abs(avg_return))
        else:
            consistency_score = 0
            
        print(f"\n🎯 ОЦЕНКА СТАБИЛЬНОСТИ:")
        print(f"📊 Индекс стабильности: {consistency_score:.1f}/100")
        
        if consistency_score > 70:
            assessment = "ОТЛИЧНО"
            print("✅ Стратегия показывает высокую стабильность")
        elif consistency_score > 50:
            assessment = "ХОРОШО"
            print("🟡 Стратегия показывает умеренную стабильность")
        else:
            assessment = "ТРЕБУЕТ ДОРАБОТКИ"
            print("❌ Стратегия нестабильна")
        
        print(f"\n🎓 ИТОГОВАЯ ОЦЕНКА: {assessment}")
        
        # Сохраняем результаты
        with open('real_walk_forward_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Результаты сохранены в real_walk_forward_results.json")

def main():
    """Основная функция"""
    print("🚀 РЕАЛЬНОЕ WALK FORWARD ТЕСТИРОВАНИЕ ETH СТРАТЕГИИ")
    print("="*60)
    
    # Создаем тестер
    tester = SimpleWalkForwardTester(
        data_path="../ETHUSDT-15m-2018-2025.csv",
        train_months=6,
        test_months=1,
        initial_balance=10000
    )
    
    # Запускаем тестирование
    results = tester.run_walk_forward_test()
    
    print(f"\n✅ Реальное walk forward тестирование завершено!")
    print(f"📊 Протестировано {len(results)} периодов на реальных данных ETH")

if __name__ == "__main__":
    main()