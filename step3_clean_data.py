#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 3: ОЧИСТКА ДАННЫХ (Готовим "еду" для модели)      ║
╚══════════════════════════════════════════════════════════╝

Запуск: python3 step3_clean_data.py

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ:
  Берёт сырые данные (raw_calendar.csv) и превращает их
  в чистые числа, которые модель может понять.

ЗАЧЕМ:
  Модель не понимает текст "0.4%" или "150K".
  Ей нужны чистые числа: 0.4 или 150000.

  Это как перевести книгу с иностранного языка перед
  тем, как дать её студенту.

РЕЗУЛЬТАТ:
  Файл data/clean_calendar.csv — чистые данные.
"""
import pandas as pd
import numpy as np
import os

# ═══════════════════════════════════════════════════════
# 📚 СЛОВАРИК:
#
# pandas — библиотека для работы с таблицами в Python
# DataFrame — это таблица (как в Excel)
# NaN — "нет данных" (пустая ячейка)
# parse — разобрать текст и извлечь из него числа
# ═══════════════════════════════════════════════════════

INPUT_FILE = "data/raw_calendar.csv"
OUTPUT_FILE = "data/clean_calendar.csv"


def parse_value(val):
    """
    Превращает текст в число.
    
    Примеры:
      "0.4%"  → 0.4     (убираем знак процента)
      "150K"  → 150000  (K = тысяча)
      "2.5M"  → 2500000 (M = миллион)
      "1.2B"  → 1200000000 (B = миллиард)
      "1,234" → 1234    (убираем запятые)
      ""      → NaN     (пусто = нет данных)
    """
    # Если пусто — вернуть NaN (нет данных)
    if pd.isna(val) or val == '' or val is None:
        return np.nan
    
    # Убираем пробелы и запятые
    val = str(val).strip().replace(',', '')
    
    # Если есть знак % — просто убираем его
    # Почему? Потому что 0.4% = число 0.4 для модели
    if '%' in val:
        try:
            return float(val.replace('%', ''))
        except:
            return np.nan
    
    # Если есть суффикс K, M, B — умножаем
    suffix_map = {
        'K': 1_000,          # Тысяча (Kilo)
        'M': 1_000_000,      # Миллион (Mega)
        'B': 1_000_000_000,  # Миллиард (Billion)
    }
    
    last_char = val[-1].upper() if val else ''
    if last_char in suffix_map:
        try:
            number = float(val[:-1])        # Число без буквы
            multiplier = suffix_map[last_char]  # Множитель
            return number * multiplier       # Результат
        except:
            return np.nan
    
    # Просто число
    try:
        return float(val)
    except:
        return np.nan


def determine_unit_type(event_name):
    """
    Определяет ТИП события по его названию.
    
    Зачем?
      "Розничные продажи (м/м)" — это ПРОЦЕНТЫ (0.1 - 5.0)
      "ВВП" — это ВАЛЮТА (триллионы $)
      "PMI" — это ИНДЕКС (40 - 60)
    
    Модель путается, если смешать проценты и триллионы.
    Поэтому мы учим ОТДЕЛЬНУЮ модель для каждого типа.
    """
    event_lower = str(event_name).lower()
    
    # Индексы (значения 0-100)
    if any(x in event_lower for x in ['index', 'pmi', 'confidence', 'sentiment', 'ism', 'survey']):
        return 'Index'
    
    # Проценты (значения -5 до +10)
    if any(x in event_lower for x in ['%', '(mom)', '(yoy)', 'rate', 'cpi', 'ppi', 'inflation', 'yield']):
        return 'Percent'
    
    # Рабочие места (значения в тысячах)
    if any(x in event_lower for x in ['jobless', 'payroll', 'employment', 'jobs']):
        return 'Jobs'
    
    # Всё остальное — валюта/деньги (миллиарды)
    return 'Currency'


def clean_data():
    """Главная функция очистки."""
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  🧹 ОЧИСТКА ДАННЫХ                              ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # 1. Загружаем сырые данные
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден!")
        print("   Сначала запустите: python3 step2_collect_data.py")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"\n📦 Загружено строк: {len(df)}")
    print(f"   Колонки: {list(df.columns)}")
    
    # Адаптируем формат колонок
    if 'Impact' in df.columns and 'Importance' not in df.columns:
        # Конвертируем Impact в числовую важность (1-3)
        impact_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Holiday': 0, 'Non-Economic': 0}
        df['Importance'] = df['Impact'].map(impact_map).fillna(1).astype(int)
    
    # Убираем нерелевантные события (праздники и т.д.)
    if 'Impact' in df.columns:
        df = df[~df['Impact'].isin(['Non-Economic', 'Holiday'])]
        print(f"   🗑 Убраны праздники. Осталось: {len(df)}")
    
    # 2. Показываем пример СЫРЫХ данных
    print(f"\n📋 Пример СЫРЫХ данных (до очистки):")
    print(df[['Event', 'Actual', 'Forecast', 'Previous']].head(3).to_string())
    
    # 3. Парсим дату
    print("\n⏰ Парсим даты...")
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['DateTime'])
    
    # 4. Парсим числа (САМЫЙ ВАЖНЫЙ ШАГ!)
    print("🔢 Парсим числовые значения...")
    print("   Actual:   '0.4%' → 0.4")
    print("   Forecast: '150K' → 150000")
    print("   Previous: '2.5M' → 2500000")
    
    for col in ['Actual', 'Forecast', 'Previous']:
        df[f'{col}_Raw'] = df[col]  # Сохраняем оригинал
        df[col] = df[col].apply(parse_value)
    
    # 5. Заполняем пустую валюту
    df['Currency'] = df['Currency'].fillna('USD')
    
    # 6. Определяем тип события
    df['Unit_Type'] = df['Event'].apply(determine_unit_type)
    
    # 7. Убираем строки без Факта (нечего предсказывать)
    before = len(df)
    df = df.dropna(subset=['Actual'])
    after = len(df)
    print(f"\n🗑 Убрано {before - after} строк без Факта (нечего предсказывать)")
    
    # 8. Сортируем по дате
    df = df.sort_values('DateTime')
    
    # 9. Сохраняем
    df.to_csv(OUTPUT_FILE, index=False)
    
    # 10. Показываем результат
    print(f"""
╔══════════════════════════════════════════════════╗
║  ✅ ОЧИСТКА ЗАВЕРШЕНА!                           ║
╚══════════════════════════════════════════════════╝

  📁 Файл: {OUTPUT_FILE}
  📊 Строк: {len(df)}
  
  📋 Пример ЧИСТЫХ данных:
""")
    print(df[['DateTime', 'Event', 'Unit_Type', 'Actual', 'Forecast', 'Previous']].head(5).to_string())
    
    print(f"""
  📊 Распределение по типам:
""")
    for unit, count in df['Unit_Type'].value_counts().items():
        bar = "█" * (count // 200)
        print(f"     {unit:10}: {count:5} событий {bar}")
    
    print(f"\n  Следующий шаг: python3 step4_train_model.py")


if __name__ == "__main__":
    clean_data()
