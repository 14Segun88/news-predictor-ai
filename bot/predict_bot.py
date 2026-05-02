#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 6: ИНТЕРАКТИВНЫЙ БОТ (Спрашивай — получай ответ)  ║
╚══════════════════════════════════════════════════════════╝

Запуск: python3 step6_predict_bot.py

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ:
  Загружает обученные модели и ждёт твоих вопросов.
  Ты вводишь данные новости — бот предсказывает Факт.

КАК ПОЛЬЗОВАТЬСЯ:
  1. Запусти скрипт
  2. Бот спросит: "Событие?" — введи название
  3. Бот спросит: "Прогноз?" — введи число
  4. Бот спросит: "Пред. значение?" — введи число
  5. Бот ответит: Факт + Уверенность

  Для выхода: 'выход' или 'quit'
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════
# 📚 СЛОВАРИК:
#
# Inference — процесс предсказания (модель уже обучена)
# Confidence — уверенность модели (0-100%)
# Routing — выбор правильной модели по типу события
# ═══════════════════════════════════════════════════════

MODELS_DIR = "models"
DATA_FILE = "data/clean_calendar.csv"

# Словарь: Русские названия → Английские (с investing.com)
EVENT_MAP = {
    "розничные продажи": "Retail Sales (MoM)",
    "объём розничных продаж": "Retail Sales (MoM)",
    "ввп": "GDP (QoQ)",
    "ставка фрс": "Fed Interest Rate Decision",
    "процентная ставка": "Fed Interest Rate Decision",
    "нон-фарм": "Nonfarm Payrolls",
    "nonfarm payrolls": "Nonfarm Payrolls",
    "nonfarm": "Nonfarm Payrolls",
    "ism pmi": "ISM Manufacturing PMI",
    "pmi": "ISM Manufacturing PMI",
    "cpi": "CPI (MoM)",
    "инфляция": "CPI (YoY)",
    "безработица": "Unemployment Rate",
    "заявки по безработице": "Initial Jobless Claims",
    "доверие потребителей": "CB Consumer Confidence",
    "торговый баланс": "Trade Balance",
    "ppi": "PPI (MoM)",
}


def determine_unit_type(event_name):
    """Определяет тип события."""
    event_lower = str(event_name).lower()
    if any(x in event_lower for x in ['index', 'pmi', 'confidence', 'sentiment', 'ism', 'survey']):
        return 'Index'
    if any(x in event_lower for x in ['%', '(mom)', '(yoy)', 'rate', 'cpi', 'ppi', 'inflation', 'yield']):
        return 'Percent'
    if any(x in event_lower for x in ['jobless', 'payroll', 'employment', 'jobs']):
        return 'Jobs'
    return 'Currency'


def parse_input(val_str):
    """Парсит ввод пользователя: '0,4%' → 0.4"""
    val = val_str.strip().replace(',', '.').replace('%', '').replace(' ', '')
    suffix_map = {'k': 1e3, 'K': 1e3, 'M': 1e6, 'B': 1e9}
    multiplier = 1
    if val and val[-1] in suffix_map:
        multiplier = suffix_map[val[-1]]
        val = val[:-1]
    try:
        return float(val) * multiplier
    except:
        return None


def resolve_event(user_input):
    """Переводит русское название в английское."""
    lower = user_input.strip().lower()
    for key, val in EVENT_MAP.items():
        if key in lower or lower in key:
            return val
    return user_input.strip()


class PredictionBot:
    """
    Бот-предсказатель.
    
    Как работает:
    1. При запуске загружает 4 модели (Percent, Index, Jobs, Currency)
    2. При вопросе:
       a) Определяет тип события
       b) Выбирает нужную модель
       c) Предсказывает Факт
       d) Рассчитывает уверенность
    """
    
    def __init__(self):
        self.models = {}
        self.event_stats = {}
    
    def load(self):
        """Загружает модели и статистику."""
        print("   📦 Загрузка моделей...")
        for unit in ['percent', 'index', 'jobs', 'currency']:
            path = f"{MODELS_DIR}/catboost_{unit}.cbm"
            if os.path.exists(path):
                model = CatBoostRegressor()
                model.load_model(path)
                self.models[unit.capitalize()] = model
                print(f"      ✅ {unit}")
            else:
                print(f"      ⚠️ Не найдена: {unit}")
        
        # Загружаем статистику для расчёта уверенности
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            train = df[df['DateTime'].dt.year < 2020]
            for event in train['Event'].unique():
                edf = train[train['Event'] == event]
                if len(edf) >= 3:
                    self.event_stats[event] = {
                        'mean': edf['Actual'].mean(),
                        'std': edf['Actual'].std() + 1e-8,
                        'count': len(edf)
                    }
    
    def predict(self, event_name, currency, importance, forecast, previous):
        """
        Предсказывает Факт.
        
        Возвращает: (предсказание, уверенность, тип_события)
        """
        unit = determine_unit_type(event_name)
        model = self.models.get(unit)
        
        if not model:
            return None, 0, f"Нет модели для: {unit}"
        
        now = datetime.now()
        prev_log = np.sign(previous) * np.log10(abs(previous) + 1)
        fore_log = np.sign(forecast) * np.log10(abs(forecast) + 1)
        
        feat = [currency, event_name, importance, forecast, previous,
                now.year, now.month, now.day, now.hour, now.weekday(),
                prev_log, fore_log]
        
        raw = model.predict([feat])
        
        if len(raw.shape) > 1 and raw.shape[1] >= 2:
            pred = raw[0, 0]
            variance = max(raw[0, 1], 0)
            std = np.sqrt(variance)
        else:
            pred = raw[0] if len(raw.shape) == 1 else raw[0, 0]
            std = 0
        
        # Уверенность
        scale = max(abs(previous), abs(forecast), 0.01)
        stats = self.event_stats.get(event_name)
        
        if stats and stats['count'] >= 5:
            rel_std = std / scale
            conf_model = max(10, min(95, 95 - rel_std * 60))
            hist_spread = stats['std'] / scale
            conf_hist = max(10, min(95, 95 - hist_spread * 20))
            confidence = round(0.6 * conf_model + 0.4 * conf_hist, 1)
        else:
            confidence = 35.0
        
        confidence = min(95, max(5, confidence))
        return pred, confidence, unit


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║  🤖 ПРЕДСКАЗАТЕЛЬ ЭКОНОМИЧЕСКИХ НОВОСТЕЙ               ║
║                                                        ║
║  Введи данные новости — получи прогноз Факта           ║
║  и уверенность модели (0-100%)                         ║
╚══════════════════════════════════════════════════════════╝
""")
    
    bot = PredictionBot()
    bot.load()
    
    if not bot.models:
        print("❌ Модели не найдены! Сначала: python3 step4_train_model.py")
        return
    
    print("""
╔══════════════════════════════════════════════════════════╗
║  ✅ Бот готов! Вводи данные новости.                    ║
║                                                        ║
║  💡 Можно писать на русском:                            ║
║     "розничные продажи", "безработица", "cpi"           ║
║                                                        ║
║  🚪 Для выхода: выход / quit                            ║
╚══════════════════════════════════════════════════════════╝
""")
    
    while True:
        print("─" * 55)
        
        # 1. Событие
        event_input = input("📌 Событие: ").strip()
        if event_input.lower() in ['quit', 'exit', 'выход', 'q', '']:
            print("\n👋 До свидания!")
            break
        
        event_en = resolve_event(event_input)
        if event_en != event_input:
            print(f"   → Найдено: {event_en}")
        
        # 2. Валюта
        currency = input("💱 Валюта (USD/RUB) [USD]: ").strip().upper()
        if not currency:
            currency = "USD"
        
        # 3. Важность
        imp = input("⭐ Важность (1-3) [3]: ").strip()
        importance = int(imp) if imp.isdigit() else 3
        
        # 4. Прогноз
        forecast_str = input("📊 Прогноз: ").strip()
        forecast = parse_input(forecast_str)
        if forecast is None:
            print("❌ Не понял число. Пример: 0.4 или 0,4%")
            continue
        
        # 5. Предыдущее
        prev_str = input("📈 Пред. значение: ").strip()
        previous = parse_input(prev_str)
        if previous is None:
            print("❌ Не понял число. Пример: 0.6 или 150K")
            continue
        
        # 6. ПРЕДСКАЗАНИЕ!
        pred, conf, unit = bot.predict(event_en, currency, importance, forecast, previous)
        
        if pred is None:
            print(f"❌ {unit}")
            continue
        
        # Красивый вывод
        bar_len = int(conf / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        if pred > forecast:
            direction = "📈 Факт ВЫШЕ прогноза (позитивный сюрприз)"
        elif pred < forecast:
            direction = "📉 Факт НИЖЕ прогноза (негативный сюрприз)"
        else:
            direction = "➡️ Факт ≈ прогноз"
        
        print(f"""
╔══════════════════════════════════════════════════════════╗
║  🔮 ПРЕДСКАЗАНИЕ МОДЕЛИ                                ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║  📌 Событие:    {event_en:40} ║
║  📊 Прогноз:    {forecast:<40} ║
║  📈 Пред.:      {previous:<40} ║
║                                                        ║
║  ➡️  Факт:       {pred:<40.4f} ║
║  🎯 Уверенность: {conf}% {bar:20}           ║
║                                                        ║
║  {direction:54} ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
