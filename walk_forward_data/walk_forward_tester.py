#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 WALK FORWARD ТЕСТИРОВАНИЕ
Система для валидации торговой стратегии с помощью walk forward анализа
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

# Добавляем родительскую папку в путь для импорта стратегии
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from balanced_strategy_base import BalancedAdaptiveStrategy

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

@dataclass
class WalkForwardResult:
    """Результат одного периода walk forward теста"""
    period_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_records: int
    test_records: int
    total_return: float
    monthly_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    total_trades: int
    best_params: Dict
    performance_metrics: Dict

class WalkForwardTester:
    """
    Система Walk Forward тестирования для торговых стратегий
    
    Принцип работы:
    1. Разделяем данные на периоды (например, 6 месяцев обучение + 1 месяц тест)
    2. Для каждого периода: оптимизируем параметры на обучающих данных
    3. Тестируем найденные параметры на следующем периоде
    4. Сдвигаем окно и повторяем
    """
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "walk_forward_data",
                 train_months: int = 6,
                 test_months: int = 1,
                 min_data_points: int = 5000,
                 initial_balance: float = 10000):
        """
        Args:
            data_path: Путь к файлу с данными
            output_dir: Директория для сохранения результатов
            train_months: Количество месяцев для обучения
            test_months: Количество месяцев для тестирования
            min_data_points: Минимальное количество точек данных для периода
            initial_balance: Начальный баланс для тестирования
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.train_months = train_months
        self.test_months = test_months
        self.min_data_points = min_data_points
        self.initial_balance = initial_balance
        
        # Создаем директории
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/periods", exist_ok=True)
        os.makedirs(f"{self.output_dir}/results", exist_ok=True)
        
        # Загружаем данные
        self.data = self._load_and_prepare_data()
        self.periods = self._create_periods()
        self.results: List[WalkForwardResult] = []
        
        print(f"🚀 Walk Forward Tester инициализирован")
        print(f"📊 Данных загружено: {len(self.data):,} записей")
        print(f"📅 Период данных: {self.data.index[0]} - {self.data.index[-1]}")
        print(f"🔄 Периодов для тестирования: {len(self.periods)}")
        print(f"📈 Обучение: {self.train_months} мес, Тест: {self.test_months} мес")
    
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Загружаем и подготавливаем данные"""
        print("📊 Загружаем данные...")
        
        df = pd.read_csv(self.data_path)
        
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
            if df["Open time"].max() < 1e12:
                # Likely seconds
                df["Open time"] = pd.to_datetime(df["Open time"], unit="s", errors="coerce")
            else:
                df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", errors="coerce")
        else:
            df["Open time"] = pd.to_datetime(df["Open time"], errors="coerce")
        
        df.dropna(subset=["Open time"], inplace=True)
        df.set_index("Open time", inplace=True)
        
        # Конвертируем цены в числовой формат
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.dropna(inplace=True)
        
        # Сортируем по времени
        df.sort_index(inplace=True)
        
        print(f"✅ Данные подготовлены: {len(df):,} записей")
        return df
    
    def _create_periods(self) -> List[Tuple[str, str, str, str]]:
        """Создаем периоды для walk forward тестирования"""
        periods = []
        
        # Начинаем с того момента, когда у нас есть достаточно данных для обучения
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        
        current_date = start_date
        period_id = 1
        
        while True:
            # Период обучения
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_months)
            
            # Период тестирования  
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            # Проверяем, что у нас достаточно данных
            if test_end > end_date:
                break
                
            # Проверяем количество записей в каждом периоде
            train_data = self.data[train_start:train_end]
            test_data = self.data[test_start:test_end] 
            
            if len(train_data) < self.min_data_points or len(test_data) < 100:
                print(f"⚠️  Период {period_id}: недостаточно данных (train: {len(train_data)}, test: {len(test_data)})")
                current_date = current_date + pd.DateOffset(months=self.test_months)
                period_id += 1
                continue
            
            periods.append((
                train_start.strftime('%Y-%m-%d %H:%M:%S'),
                train_end.strftime('%Y-%m-%d %H:%M:%S'),
                test_start.strftime('%Y-%m-%d %H:%M:%S'),
                test_end.strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            # Сдвигаем окно
            current_date = current_date + pd.DateOffset(months=self.test_months)
            period_id += 1
        
        print(f"🔄 Создано {len(periods)} периодов для тестирования")
        return periods
    
    def _save_period_data(self, period_id: int, train_data: pd.DataFrame, 
                         test_data: pd.DataFrame) -> Tuple[str, str]:
        """Сохраняем данные периода в отдельные файлы"""
        train_path = f"{self.output_dir}/periods/period_{period_id:03d}_train.csv"
        test_path = f"{self.output_dir}/periods/period_{period_id:03d}_test.csv"
        
        # Подготавливаем данные для сохранения (возвращаем к исходному формату)
        def prepare_for_save(df):
            df_save = df.copy()
            df_save.reset_index(inplace=True)
            df_save.rename(columns={
                "Open time": "open_time",
                "Open": "open",
                "High": "high",
                "Low": "low", 
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            # Convert datetime back to timestamp (milliseconds)
            df_save['open_time'] = (pd.to_datetime(df_save['open_time']).astype('int64') // 10**6)
            return df_save
        
        prepare_for_save(train_data).to_csv(train_path, index=False)
        prepare_for_save(test_data).to_csv(test_path, index=False)
        
        return train_path, test_path
    
    def _optimize_on_training_data(self, train_path: str, period_id: int) -> Dict:
        """Оптимизация параметров на обучающих данных"""
        print(f"🔧 Период {period_id}: Оптимизация параметров...")
        
        try:
            # Создаем стратегию для оптимизации
            strategy = BalancedAdaptiveStrategy(
                data_path=train_path,
                symbol="ETH",
                initial_balance=self.initial_balance,
                max_leverage=10,
                base_risk_per_trade=0.02
            )
            
            # Параметры для оптимизации
            param_ranges = {
                'global_long_boost': [1.1, 1.2, 1.3, 1.4, 1.5],
                'global_short_penalty': [0.2, 0.3, 0.4, 0.5, 0.6],
                'atr_multiplier_sl': [1.5, 2.0, 2.5, 3.0],
                'atr_multiplier_tp': [4.0, 5.0, 6.0, 7.0, 8.0],
                'long_entry_threshold': [0.55, 0.60, 0.65, 0.70],
                'short_entry_threshold': [0.70, 0.75, 0.80, 0.85],
                'min_trades_interval': [4, 6, 8, 10, 12]
            }
            
            # Запускаем оптимизацию (ограниченную для скорости)
            best_result = strategy.optimize_parameters(param_ranges, num_trials=30)
            
            print(f"✅ Период {period_id}: Оптимизация завершена")
            
            # Возвращаем результат или пустой словарь, если оптимизация не удалась
            if best_result is not None:
                return best_result
            else:
                return {'best_params': {}, 'best_score': 0}
                
        except Exception as e:
            print(f"❌ Ошибка оптимизации в периоде {period_id}: {str(e)}")
            return {'best_params': {}, 'best_score': 0}
    
    def _test_on_forward_data(self, test_path: str, best_params: Dict, 
                             period_id: int) -> Dict:
        """Тестирование найденных параметров на forward данных"""
        print(f"📊 Период {period_id}: Тестирование на forward данных...")
        
        try:
            # Создаем стратегию с оптимальными параметрами
            strategy = BalancedAdaptiveStrategy(
                data_path=test_path,
                symbol="ETH",
                initial_balance=self.initial_balance,
                max_leverage=10,
                base_risk_per_trade=0.02
            )
            
            # Применяем найденные параметры
            strategy.params.update(best_params)
            
            # Запускаем бэктест
            strategy.load_data()
            strategy._auto_scale_volatility()
            strategy.calculate_indicators()
            strategy.run_backtest()
            results = strategy.analyze_results()
            
            print(f"✅ Период {period_id}: Тестирование завершено")
            
            # Возвращаем результат или пустой словарь с нулевыми значениями
            if results is not None:
                return results
            else:
                return {
                    'total_return': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'total_trades': 0
                }
                
        except Exception as e:
            print(f"❌ Ошибка тестирования в периоде {period_id}: {str(e)}")
            return {
                'total_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'total_trades': 0
            }
    
    def run_walk_forward_test(self) -> List[WalkForwardResult]:
        """Запуск полного walk forward тестирования"""
        print("\n" + "="*60)
        print("🚀 ЗАПУСК WALK FORWARD ТЕСТИРОВАНИЯ")
        print("="*60)
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(self.periods, 1):
            print(f"\n📅 ПЕРИОД {i}/{len(self.periods)}")
            print(f"🔄 Обучение: {train_start} - {train_end}")
            print(f"📊 Тестирование: {test_start} - {test_end}")
            
            try:
                # Извлекаем данные для периода
                train_data = self.data[train_start:train_end]
                test_data = self.data[test_start:test_end]
                
                print(f"📈 Данных для обучения: {len(train_data):,}")
                print(f"📉 Данных для тестирования: {len(test_data):,}")
                
                # Сохраняем данные периода
                train_path, test_path = self._save_period_data(i, train_data, test_data)
                
                # Оптимизируем параметры на обучающих данных
                optimization_result = self._optimize_on_training_data(train_path, i)
                best_params = optimization_result.get('best_params', {})
                
                # Тестируем на forward данных
                test_results = self._test_on_forward_data(test_path, best_params, i)
                
                # Создаем результат периода
                monthly_return = test_results.get('monthly_return', 
                                                test_results['total_return'] / self.test_months)
                
                period_result = WalkForwardResult(
                    period_id=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_records=len(train_data),
                    test_records=len(test_data),
                    total_return=test_results['total_return'],
                    monthly_return=monthly_return,
                    max_drawdown=test_results['max_drawdown'],
                    win_rate=test_results['win_rate'],
                    profit_factor=test_results.get('profit_factor', 0),
                    sharpe_ratio=test_results.get('sharpe_ratio', 0),
                    total_trades=test_results['total_trades'],
                    best_params=best_params,
                    performance_metrics=test_results
                )
                
                self.results.append(period_result)
                
                # Выводим результаты периода
                print(f"💰 Доходность: {period_result.total_return:.2f}%")
                print(f"📈 Месячная доходность: {period_result.monthly_return:.2f}%")
                print(f"📉 Макс. просадка: {period_result.max_drawdown:.2f}%")
                print(f"🎯 Win Rate: {period_result.win_rate:.1f}%")
                print(f"🔄 Сделок: {period_result.total_trades}")
                
            except Exception as e:
                print(f"❌ Ошибка в периоде {i}: {str(e)}")
                continue
        
        # Сохраняем результаты
        self._save_results()
        
        print(f"\n✅ Walk Forward тестирование завершено!")
        print(f"📊 Протестировано периодов: {len(self.results)}")
        
        return self.results
    
    def _save_results(self):
        """Сохранение результатов в файлы"""
        # Сохраняем детальные результаты в JSON
        results_data = []
        for result in self.results:
            results_data.append({
                'period_id': result.period_id,
                'train_start': result.train_start,
                'train_end': result.train_end,
                'test_start': result.test_start,
                'test_end': result.test_end,
                'train_records': result.train_records,
                'test_records': result.test_records,
                'total_return': result.total_return,
                'monthly_return': result.monthly_return,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'total_trades': result.total_trades,
                'best_params': result.best_params,
                'performance_metrics': result.performance_metrics
            })
        
        with open(f"{self.output_dir}/results/walk_forward_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Сохраняем сводную таблицу в CSV
        df_results = pd.DataFrame([
            {
                'Period': r.period_id,
                'Test_Start': r.test_start,
                'Test_End': r.test_end,
                'Total_Return_%': r.total_return,
                'Monthly_Return_%': r.monthly_return,
                'Max_Drawdown_%': r.max_drawdown,
                'Win_Rate_%': r.win_rate,
                'Profit_Factor': r.profit_factor,
                'Sharpe_Ratio': r.sharpe_ratio,
                'Total_Trades': r.total_trades
            }
            for r in self.results
        ])
        
        df_results.to_csv(f"{self.output_dir}/results/summary_results.csv", index=False)
        print(f"💾 Результаты сохранены в {self.output_dir}/results/")
    
    def analyze_walk_forward_results(self) -> Dict:
        """Анализ результатов walk forward тестирования"""
        if not self.results:
            print("❌ Нет результатов для анализа")
            return {}
        
        print("\n" + "="*60)
        print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ WALK FORWARD ТЕСТИРОВАНИЯ")
        print("="*60)
        
        # Собираем метрики
        monthly_returns = [r.monthly_return for r in self.results]
        total_returns = [r.total_return for r in self.results]
        max_drawdowns = [r.max_drawdown for r in self.results]
        win_rates = [r.win_rate for r in self.results]
        profit_factors = [r.profit_factor for r in self.results if r.profit_factor > 0]
        sharpe_ratios = [r.sharpe_ratio for r in self.results if not np.isnan(r.sharpe_ratio)]
        total_trades = [r.total_trades for r in self.results]
        
        # Основная статистика
        stats = {
            'periods_tested': len(self.results),
            'avg_monthly_return': np.mean(monthly_returns),
            'median_monthly_return': np.median(monthly_returns),
            'std_monthly_return': np.std(monthly_returns),
            'min_monthly_return': np.min(monthly_returns),
            'max_monthly_return': np.max(monthly_returns),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'max_max_drawdown': np.max(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'avg_profit_factor': np.mean(profit_factors) if profit_factors else 0,
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'total_trades_avg': np.mean(total_trades),
            'profitable_periods': len([r for r in monthly_returns if r > 0]),
            'profitable_periods_pct': len([r for r in monthly_returns if r > 0]) / len(monthly_returns) * 100
        }
        
        # Выводим статистику
        print(f"📊 Протестировано периодов: {stats['periods_tested']}")
        print(f"📈 Средняя месячная доходность: {stats['avg_monthly_return']:.2f}%")
        print(f"📈 Медианная месячная доходность: {stats['median_monthly_return']:.2f}%")
        print(f"📈 Волатильность доходности: {stats['std_monthly_return']:.2f}%")
        print(f"📈 Диапазон доходности: {stats['min_monthly_return']:.2f}% - {stats['max_monthly_return']:.2f}%")
        print(f"📉 Средняя макс. просадка: {stats['avg_max_drawdown']:.2f}%")
        print(f"📉 Максимальная просадка: {stats['max_max_drawdown']:.2f}%")
        print(f"🎯 Средний Win Rate: {stats['avg_win_rate']:.1f}%")
        print(f"💰 Средний Profit Factor: {stats['avg_profit_factor']:.2f}")
        print(f"📊 Средний Sharpe Ratio: {stats['avg_sharpe_ratio']:.2f}")
        print(f"✅ Прибыльных периодов: {stats['profitable_periods']} ({stats['profitable_periods_pct']:.1f}%)")
        
        # Расчет годовой доходности с учетом волатильности
        annual_return = stats['avg_monthly_return'] * 12
        annual_volatility = stats['std_monthly_return'] * np.sqrt(12)
        
        print(f"\n📅 ГОДОВЫЕ ПОКАЗАТЕЛИ:")
        print(f"📈 Ожидаемая годовая доходность: {annual_return:.1f}%")
        print(f"📊 Годовая волатильность: {annual_volatility:.1f}%")
        print(f"📊 Риск-скорректированная доходность: {annual_return/annual_volatility:.2f}")
        
        # Сравнение с бенчмарком
        print(f"\n🏆 СРАВНЕНИЕ С РЫНКОМ:")
        sp500_annual = 10  # Средняя доходность S&P 500
        btc_annual = 50   # Средняя доходность Bitcoin
        
        print(f"📊 Наша стратегия: {annual_return:.1f}% годовых")
        print(f"📊 S&P 500: ~{sp500_annual}% годовых ({'превосходим' if annual_return > sp500_annual else 'уступаем'} в {abs(annual_return/sp500_annual):.1f}x раз)")
        print(f"📊 Bitcoin: ~{btc_annual}% годовых ({'превосходим' if annual_return > btc_annual else 'уступаем'} в {abs(annual_return/btc_annual):.1f}x раз)")
        
        # Стабильность результатов
        if stats['avg_monthly_return'] != 0:
            consistency_score = stats['profitable_periods_pct'] * (1 - stats['std_monthly_return'] / abs(stats['avg_monthly_return']))
        else:
            consistency_score = 0
        print(f"\n🎯 ОЦЕНКА СТАБИЛЬНОСТИ:")
        print(f"📊 Индекс стабильности: {consistency_score:.1f}/100")
        
        if consistency_score > 70:
            print("✅ Стратегия показывает высокую стабильность")
        elif consistency_score > 50:
            print("🟡 Стратегия показывает умеренную стабильность")
        else:
            print("❌ Стратегия нестабильна")
        
        # Сохраняем анализ
        with open(f"{self.output_dir}/results/walk_forward_analysis.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def plot_walk_forward_results(self):
        """Построение графиков результатов walk forward тестирования"""
        if not self.results:
            print("❌ Нет результатов для построения графиков")
            return
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Walk Forward Testing Results', fontsize=16, fontweight='bold')
        
        # График 1: Месячная доходность по периодам
        periods = [r.period_id for r in self.results]
        monthly_returns = [r.monthly_return for r in self.results]
        
        axes[0, 0].bar(periods, monthly_returns, 
                      color=['green' if x > 0 else 'red' for x in monthly_returns],
                      alpha=0.7)
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].axhline(y=np.mean(monthly_returns), color='blue', 
                          linestyle='--', alpha=0.7, label=f'Среднее: {np.mean(monthly_returns):.2f}%')
        axes[0, 0].set_title('Месячная доходность по периодам')
        axes[0, 0].set_xlabel('Период')
        axes[0, 0].set_ylabel('Доходность (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Максимальная просадка по периодам
        max_drawdowns = [r.max_drawdown for r in self.results]
        
        axes[0, 1].bar(periods, max_drawdowns, color='red', alpha=0.7)
        axes[0, 1].axhline(y=np.mean(max_drawdowns), color='blue', 
                          linestyle='--', alpha=0.7, label=f'Среднее: {np.mean(max_drawdowns):.2f}%')
        axes[0, 1].set_title('Максимальная просадка по периодам')
        axes[0, 1].set_xlabel('Период')
        axes[0, 1].set_ylabel('Просадка (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Win Rate по периодам
        win_rates = [r.win_rate for r in self.results]
        
        axes[1, 0].plot(periods, win_rates, marker='o', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Breakeven (50%)')
        axes[1, 0].axhline(y=np.mean(win_rates), color='blue', 
                          linestyle='--', alpha=0.7, label=f'Среднее: {np.mean(win_rates):.1f}%')
        axes[1, 0].set_title('Win Rate по периодам')
        axes[1, 0].set_xlabel('Период')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Распределение доходности
        axes[1, 1].hist(monthly_returns, bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].axvline(x=np.mean(monthly_returns), color='red', 
                          linestyle='--', alpha=0.7, label=f'Среднее: {np.mean(monthly_returns):.2f}%')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Распределение месячной доходности')
        axes[1, 1].set_xlabel('Месячная доходность (%)')
        axes[1, 1].set_ylabel('Количество периодов')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/results/walk_forward_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Графики сохранены в {self.output_dir}/results/walk_forward_analysis.png")

def main():
    """Основная функция для запуска walk forward тестирования"""
    print("🚀 ETH WALK FORWARD ТЕСТИРОВАНИЕ")
    print("="*50)
    
    # Настройки тестирования
    data_path = "../ETHUSDT-15m-2018-2025.csv"
    
    # Создаем тестер
    tester = WalkForwardTester(
        data_path=data_path,
        output_dir="walk_forward_data",
        train_months=6,  # 6 месяцев обучения
        test_months=1,   # 1 месяц тестирования  
        min_data_points=5000,
        initial_balance=10000
    )
    
    # Запускаем тестирование
    results = tester.run_walk_forward_test()
    
    # Анализируем результаты
    tester.analyze_walk_forward_results()
    
    # Строим графики
    tester.plot_walk_forward_results()
    
    print("\n✅ Walk Forward тестирование завершено!")
    print(f"📁 Результаты сохранены в папке: walk_forward_data/")

if __name__ == "__main__":
    main()