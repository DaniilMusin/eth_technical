#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 ДЕМО Walk Forward Тестирования (быстрая версия)
Упрощенная версия для демонстрации принципа работы
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
from datetime import datetime

# Добавляем родительскую папку в путь для импорта стратегии
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Загружаем и подготавливаем данные"""
    print("📊 Загружаем данные...")
    
    df = pd.read_csv(file_path)
    
    # Стандартизируем названия колонок
    rename_map = {
        "open_time": "Open time",
        "open": "Open",
        "high": "High", 
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Конвертируем время
    if df["Open time"].dtype.kind in "iu":
        # Check if timestamp is in seconds or milliseconds
        if df["Open time"].max() < 1e12:
            # Likely seconds
            df["Open time"] = pd.to_datetime(df["Open time"], unit="s", errors="coerce")
        else:
            # Likely milliseconds
            df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", errors="coerce")
    else:
        df["Open time"] = pd.to_datetime(df["Open time"], errors="coerce")
    
    df.dropna(subset=["Open time"], inplace=True)
    df.set_index("Open time", inplace=True)
    
    # Конвертируем цены в числовой формат
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    
    print(f"✅ Данные подготовлены: {len(df):,} записей")
    print(f"📅 Период: {df.index[0]} - {df.index[-1]}")
    return df

def create_simple_periods(data, train_months=3, test_months=1):
    """Создаем упрощенные периоды для демонстрации"""
    periods = []
    
    start_date = data.index[0]
    end_date = data.index[-1]
    
    current_date = start_date
    period_id = 1
    
    # Создаем только первые несколько периодов для демо
    max_periods = 5  # Ограничиваем количество для демо
    
    while period_id <= max_periods:
        # Период обучения
        train_start = current_date
        train_end = train_start + pd.DateOffset(months=train_months)
        
        # Период тестирования  
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Проверяем, что у нас достаточно данных
        if test_end > end_date:
            break
            
        # Проверяем количество записей в каждом периоде
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end] 
        
        if len(train_data) < 1000 or len(test_data) < 100:
            current_date = current_date + pd.DateOffset(months=test_months)
            period_id += 1
            continue
        
        periods.append({
            'id': period_id,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_records': len(train_data),
            'test_records': len(test_data)
        })
        
        # Сдвигаем окно
        current_date = current_date + pd.DateOffset(months=test_months)
        period_id += 1
    
    return periods

def simple_strategy_test(data):
    """Упрощенная стратегия для демонстрации"""
    # Простая стратегия на основе скользящих средних
    data = data.copy()
    
    # Рассчитываем индикаторы
    data['MA_Short'] = data['Close'].rolling(window=10).mean()
    data['MA_Long'] = data['Close'].rolling(window=30).mean()
    data['RSI'] = calculate_simple_rsi(data['Close'], 14)
    
    # Сигналы
    data['Long_Signal'] = (data['MA_Short'] > data['MA_Long']) & (data['RSI'] < 70)
    data['Short_Signal'] = (data['MA_Short'] < data['MA_Long']) & (data['RSI'] > 30)
    
    # Симуляция торговли
    balance = 10000
    position = 0
    entry_price = None
    trades = []
    
    for i in range(len(data)):
        if data['Long_Signal'].iloc[i] and position <= 0:
            if position < 0 and entry_price is not None and entry_price != 0:
                # Закрываем шорт
                pnl = (entry_price - data['Close'].iloc[i]) / entry_price * abs(position)
                balance += pnl
                trades.append({'type': 'close_short', 'pnl': pnl})
            elif position < 0:
                # entry_price is 0 or None, can't calculate PnL
                position = 0
                entry_price = None
            
            # Открываем лонг
            position = balance * 0.1  # 10% от баланса
            entry_price = data['Close'].iloc[i]
            trades.append({'type': 'open_long', 'price': entry_price})
            
        elif data['Short_Signal'].iloc[i] and position >= 0:
            if position > 0 and entry_price is not None and entry_price != 0:
                # Закрываем лонг
                pnl = (data['Close'].iloc[i] - entry_price) / entry_price * position
                balance += pnl
                trades.append({'type': 'close_long', 'pnl': pnl})
            elif position > 0:
                # entry_price is 0 or None, can't calculate PnL
                position = 0
                entry_price = None
            
            # Открываем шорт
            position = -balance * 0.1  # 10% от баланса
            entry_price = data['Close'].iloc[i]
            trades.append({'type': 'open_short', 'price': entry_price})
    
    # Финальные расчеты
    total_return = (balance - 10000) / 10000 * 100
    profitable_trades = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
    win_rate = len(profitable_trades) / max(len([t for t in trades if 'pnl' in t]), 1) * 100
    
    return {
        'total_return': total_return,
        'final_balance': balance,
        'total_trades': len([t for t in trades if 'pnl' in t]),
        'win_rate': win_rate,
        'max_drawdown': min(0, min([t.get('pnl', 0) for t in trades], default=0)) / 100
    }

def calculate_simple_rsi(prices, period):
    """Простой расчет RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def run_demo_walk_forward():
    """Запуск демонстрационного walk forward тестирования"""
    print("🚀 ДЕМО WALK FORWARD ТЕСТИРОВАНИЯ")
    print("="*50)
    print("📊 Упрощенная версия для демонстрации принципа")
    print("="*50)
    
    # Проверяем файл данных
    data_path = "../ETHUSDT-15m-2018-2025.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Файл данных не найден: {data_path}")
        return
    
    # Загружаем данные
    data = load_data(data_path)
    
    # Создаем периоды (упрощенные)
    periods = create_simple_periods(data, train_months=3, test_months=1)
    
    print(f"\n🔄 Создано {len(periods)} периодов для демонстрации")
    
    results = []
    
    # Проходим по каждому периоду
    for period in periods:
        print(f"\n📅 ПЕРИОД {period['id']}")
        print(f"🔄 Обучение: {period['train_start'].strftime('%Y-%m-%d')} - {period['train_end'].strftime('%Y-%m-%d')}")
        print(f"📊 Тестирование: {period['test_start'].strftime('%Y-%m-%d')} - {period['test_end'].strftime('%Y-%m-%d')}")
        print(f"📈 Данные: {period['train_records']:,} обучение, {period['test_records']:,} тест")
        
        # Извлекаем данные для периода
        train_data = data[period['train_start']:period['train_end']]
        test_data = data[period['test_start']:period['test_end']]
        
        # В реальном walk forward здесь была бы оптимизация на train_data
        # Для демо просто используем фиксированные параметры
        print("🔧 (В реальном тесте здесь была бы оптимизация параметров)")
        
        # Тестируем на forward данных
        result = simple_strategy_test(test_data)
        
        result['period_id'] = period['id']
        result['test_start'] = period['test_start'].strftime('%Y-%m-%d')
        result['test_end'] = period['test_end'].strftime('%Y-%m-%d')
        
        results.append(result)
        
        # Выводим результаты периода
        print(f"💰 Доходность: {result['total_return']:.2f}%")
        print(f"🎯 Win Rate: {result['win_rate']:.1f}%")
        print(f"🔄 Сделок: {result['total_trades']}")
    
    # Анализируем общие результаты
    print(f"\n" + "="*50)
    print("📊 ОБЩИЕ РЕЗУЛЬТАТЫ ДЕМО ТЕСТИРОВАНИЯ")
    print("="*50)
    
    monthly_returns = [r['total_return'] for r in results]
    
    avg_return = np.mean(monthly_returns)
    std_return = np.std(monthly_returns) 
    win_rate_avg = np.mean([r['win_rate'] for r in results])
    profitable_periods = len([r for r in monthly_returns if r > 0])
    
    print(f"📊 Протестировано периодов: {len(results)}")
    print(f"📈 Средняя месячная доходность: {avg_return:.2f}%")
    print(f"📈 Волатильность доходности: {std_return:.2f}%")
    print(f"🎯 Средний Win Rate: {win_rate_avg:.1f}%")
    print(f"✅ Прибыльных периодов: {profitable_periods} ({profitable_periods/len(results)*100:.1f}%)")
    
    # Годовая проекция
    annual_return = avg_return * 12
    print(f"\n📅 ПРОЕКЦИЯ НА ГОД:")
    print(f"📈 Ожидаемая годовая доходность: {annual_return:.1f}%")
    
    if avg_return > 0:
        print(f"✅ Стратегия показывает положительные результаты!")
    else:
        print(f"❌ Стратегия показывает отрицательные результаты")
    
    # Сохраняем результаты демо
    with open('demo_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Результаты демо сохранены в demo_results.json")
    print(f"\n💡 Это упрощенная демонстрация. Для полного анализа используйте:")
    print(f"   python run_walk_forward.py")

if __name__ == "__main__":
    run_demo_walk_forward()