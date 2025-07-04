#!/usr/bin/env python3
"""
Тестирование ультра-агрессивных конфигураций для достижения 20% в месяц
"""

import pandas as pd
import numpy as np
from balanced_strategy_base import BalancedAdaptiveStrategy
import warnings
warnings.filterwarnings('ignore')

def test_ultra_aggressive_config(config_name, **kwargs):
    """Тестирует ультра-агрессивную конфигурацию"""
    
    default_params = {
        'data_path': r"C:\programming\eth_technical\eth_technical\ETHUSDT-15m-2018-2025.csv",
        'symbol': "ETH",
        'initial_balance': 1000,
        'max_leverage': 20,
        'base_risk_per_trade': 0.04,
        'min_trades_interval': 4
    }
    
    strategy_params = kwargs.pop('strategy_params', {})
    params = {**default_params, **kwargs}
    
    print(f"\n🔥 Тестирование: {config_name}")
    print(f"⚠️  Параметры: {kwargs}")
    if strategy_params:
        print(f"🎯 Стратегия: {strategy_params}")
    
    try:
        strategy = BalancedAdaptiveStrategy(**params)
        
        # Применяем агрессивные настройки
        for key, value in strategy_params.items():
            strategy.params[key] = value
        
        strategy.load_data()
        strategy._auto_scale_volatility()
        strategy.calculate_indicators()
        strategy.run_backtest()
        stats = strategy.analyze_results()
        
        # Вычисляем месячную доходность
        monthly_return = stats.get('monthly_return', stats['total_return'] / 8.5)
        
        print(f"📊 Результат: {stats['total_return']:.2f}% общая | {monthly_return:.2f}% в месяц")
        print(f"📉 Просадка: {stats['max_drawdown']:.2f}% | Sharpe: {stats['sharpe_ratio']:.2f}")
        print(f"🎯 Win Rate: {stats['win_rate']:.2f}% | Сделок: {stats['total_trades']}")
        
        return {
            'config_name': config_name,
            'total_return': stats['total_return'],
            'monthly_return': monthly_return,
            'max_drawdown': stats['max_drawdown'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'profit_factor': stats['profit_factor'],
            'win_rate': stats['win_rate'],
            'total_trades': stats['total_trades']
        }
        
    except Exception as e:
        print(f"❌ Ошибка в {config_name}: {e}")
        return None

def main():
    """Тестирование различных ультра-агрессивных конфигураций"""
    
    results = []
    
    print("🚀 ТЕСТИРОВАНИЕ УЛЬТРА-АГРЕССИВНЫХ СТРАТЕГИЙ")
    print("🎯 Цель: 20% в месяц")
    print("⚠️  ВНИМАНИЕ: Высокий риск!")
    
    # Конфигурация 1: Умная агрессивность (высокий риск + селективность)
    result = test_ultra_aggressive_config(
        "🧠 УМНАЯ АГРЕССИВНОСТЬ",
        base_risk_per_trade=0.035,  # Умеренно высокий риск
        max_leverage=15,
        min_trades_interval=6,
        strategy_params={
            'global_long_boost': 1.40,
            'global_short_penalty': 0.20,  # Ограничиваем шорты
            'atr_multiplier_sl': 1.8,  # Не слишком тайтовые стопы
            'atr_multiplier_tp': 5.0,  # Хорошие цели
            'long_entry_threshold': 0.55,  # Селективность
            'short_entry_threshold': 0.90,  # Очень селективные шорты
        }
    )
    if result: results.append(result)
    
    # Конфигурация 2: Высокочастотная (больше сделок)
    result = test_ultra_aggressive_config(
        "⚡ ВЫСОКОЧАСТОТНАЯ",
        base_risk_per_trade=0.025,  # Меньший риск, но больше сделок
        max_leverage=12,
        min_trades_interval=3,  # Чаще торгуем
        strategy_params={
            'global_long_boost': 1.50,
            'global_short_penalty': 0.10,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 4.0,
            'long_entry_threshold': 0.45,  # Ниже порог = больше сделок
            'short_entry_threshold': 0.95,
        }
    )
    if result: results.append(result)
    
    # Конфигурация 3: Компаунд-фокус (только лучшие сигналы, высокий риск)
    result = test_ultra_aggressive_config(
        "💎 КОМПАУНД-ПРЕМИУМ",
        base_risk_per_trade=0.045,  # Высокий риск на лучшие сигналы
        max_leverage=18,
        min_trades_interval=10,  # Реже, но качественнее
        strategy_params={
            'global_long_boost': 1.60,
            'global_short_penalty': 0.05,  # Почти только лонги
            'atr_multiplier_sl': 2.2,
            'atr_multiplier_tp': 6.0,  # Высокие цели
            'long_entry_threshold': 0.70,  # Высокая селективность
            'short_entry_threshold': 0.99,  # Блокируем шорты
            'pyramid_min_profit': 0.008,  # Агрессивная пирамидинг
            'max_pyramid_entries': 4,
        }
    )
    if result: results.append(result)
    
    # Конфигурация 4: Балансированная агрессивность
    result = test_ultra_aggressive_config(
        "⚖️ СБАЛАНСИРОВАННАЯ АГРЕССИЯ",
        base_risk_per_trade=0.030,
        max_leverage=14,
        min_trades_interval=8,
        strategy_params={
            'global_long_boost': 1.35,
            'global_short_penalty': 0.30,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 5.5,
            'long_entry_threshold': 0.60,
            'short_entry_threshold': 0.80,
        }
    )
    if result: results.append(result)
    
    # Конфигурация 5: Экстремальные лонги (максимальный риск на лонги)
    result = test_ultra_aggressive_config(
        "🚀 ЛОНГИ-МАКСИМУМ",
        base_risk_per_trade=0.055,  # Очень высокий риск
        max_leverage=25,
        min_trades_interval=5,
        strategy_params={
            'global_long_boost': 1.80,  # Очень высокий буст
            'global_short_penalty': 0.01,  # Блокируем шорты
            'atr_multiplier_sl': 1.5,  # Тайтовые стопы для сохранения капитала
            'atr_multiplier_tp': 4.5,
            'long_entry_threshold': 0.50,
            'short_entry_threshold': 0.99,
        }
    )
    if result: results.append(result)
    
    # Анализ результатов
    if results:
        print("\n" + "="*80)
        print("📈 СРАВНИТЕЛЬНАЯ ТАБЛИЦА УЛЬТРА-АГРЕССИВНЫХ СТРАТЕГИЙ")
        print("="*80)
        
        df = pd.DataFrame(results)
        df = df.sort_values('monthly_return', ascending=False)
        
        for _, row in df.iterrows():
            status = ""
            if row['monthly_return'] >= 20:
                status = "🏆 ЦЕЛЬ ДОСТИГНУТА!"
            elif row['monthly_return'] >= 15:
                status = "🥈 Близко к цели"
            elif row['monthly_return'] >= 10:
                status = "🥉 Хорошо"
            else:
                status = "❌ Недостаточно"
                
            print(f"\n{row['config_name']}")
            print(f"  📊 Доходность: {row['total_return']:.1f}% общая | {row['monthly_return']:.1f}% в месяц")
            print(f"  📉 Просадка: {row['max_drawdown']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
            print(f"  🎯 Win Rate: {row['win_rate']:.1f}% | Сделок: {row['total_trades']} | {status}")
        
        # Лучший результат
        best = df.iloc[0]
        print(f"\n🏆 ЛУЧШАЯ СТРАТЕГИЯ: {best['config_name']}")
        print(f"🎯 Месячная доходность: {best['monthly_return']:.2f}%")
        
        if best['monthly_return'] >= 20:
            print("✅ МИССИЯ ВЫПОЛНЕНА! 20%+ в месяц достигнуто!")
        else:
            print(f"🔧 Нужно еще {20 - best['monthly_return']:.2f}% для достижения цели")
    
    return results

if __name__ == "__main__":
    results = main() 