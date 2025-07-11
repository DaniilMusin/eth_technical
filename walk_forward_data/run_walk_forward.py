#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ПРОСТОЙ ЗАПУСК WALK FORWARD ТЕСТИРОВАНИЯ
"""

import os
import sys

def main():
    """Простой запуск walk forward тестирования"""
    print("🚀 ЗАПУСК WALK FORWARD ТЕСТИРОВАНИЯ ETH СТРАТЕГИИ")
    print("="*60)
    print("📊 Это может занять некоторое время...")
    print("🔄 Каждый период включает оптимизацию + тестирование")
    print("="*60)
    
    try:
        # Импортируем тестер
        from walk_forward_tester import WalkForwardTester
        
        # Настройки тестирования
        data_path = "../ETHUSDT-15m-2018-2025.csv"
        
        if not os.path.exists(data_path):
            print(f"❌ Файл данных не найден: {data_path}")
            print("Убедитесь, что файл ETHUSDT-15m-2018-2025.csv находится в родительской папке")
            return
        
        # Создаем тестер с консервативными настройками
        tester = WalkForwardTester(
            data_path=data_path,
            output_dir=".",  # Сохраняем в текущую папку walk_forward_data
            train_months=6,  # 6 месяцев обучения
            test_months=1,   # 1 месяц тестирования  
            min_data_points=3000,  # Уменьшили для более частых тестов
            initial_balance=10000
        )
        
        print(f"\n⚙️  НАСТРОЙКИ ТЕСТИРОВАНИЯ:")
        print(f"📊 Обучение: {tester.train_months} месяцев")
        print(f"📈 Тестирование: {tester.test_months} месяц")
        print(f"💰 Начальный баланс: ${tester.initial_balance:,}")
        print(f"📋 Периодов для тестирования: {len(tester.periods)}")
        
        # Запускаем тестирование
        print(f"\n🔄 Начинаем walk forward тестирование...")
        results = tester.run_walk_forward_test()
        
        if results:
            # Анализируем результаты
            print(f"\n📊 Анализируем результаты...")
            stats = tester.analyze_walk_forward_results()
            
            # Строим графики
            print(f"\n📈 Строим графики...")
            tester.plot_walk_forward_results()
            
            print(f"\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print(f"📁 Все результаты сохранены в папке: walk_forward_data/")
            print(f"📊 Основные файлы:")
            print(f"   • results/summary_results.csv - Сводная таблица")
            print(f"   • results/walk_forward_results.json - Детальные результаты")
            print(f"   • results/walk_forward_analysis.png - Графики")
            print(f"   • results/walk_forward_analysis.json - Статистика")
            
        else:
            print(f"\n❌ Тестирование не дало результатов")
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Убедитесь, что все зависимости установлены:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        print("Проверьте целостность файлов и зависимостей")

if __name__ == "__main__":
    main()