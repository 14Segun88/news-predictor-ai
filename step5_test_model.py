#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 5: ТЕСТИРОВАНИЕ (Проверяем модель на "экзамене")  ║
╚══════════════════════════════════════════════════════════╝

Запуск: python3 step5_test_model.py

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ:
  Берёт данные 2020-2026, которые модель НЕ видела,
  и проверяет: правильно ли она предсказывает.

  Для каждой новости отвечает:
    ✅ Правильно — модель угадала направление
    ❌ Неправильно — модель ошиблась

АНАЛОГИЯ:
  Студент сдаёт экзамен. Мы считаем сколько
  ответов правильных, а сколько — нет.

РЕЗУЛЬТАТ:
  Файл data/test_results.csv — все предсказания.
  Метрика: % правильных ответов (Accuracy).
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score
import os

# ═══════════════════════════════════════════════════════
# 📚 СЛОВАРИК:
#
# Accuracy — точность (% правильных ответов)
# Direction — направление (Факт > Прогноз = ВВЕРХ)
# Backtest — тестирование на исторических данных
# ═══════════════════════════════════════════════════════

INPUT_FILE = "data/clean_calendar.csv"
MODELS_DIR = "models"
OUTPUT_FILE = "data/test_results.csv"


def determine_unit_type(event_name):
    """Определяет тип события (копия из step3)."""
    event_lower = str(event_name).lower()
    if any(x in event_lower for x in ['index', 'pmi', 'confidence', 'sentiment', 'ism', 'survey']):
        return 'Index'
    if any(x in event_lower for x in ['%', '(mom)', '(yoy)', 'rate', 'cpi', 'ppi', 'inflation', 'yield']):
        return 'Percent'
    if any(x in event_lower for x in ['jobless', 'payroll', 'employment', 'jobs']):
        return 'Jobs'
    return 'Currency'


def test_model():
    """Главная функция тестирования."""
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  📝 ТЕСТИРОВАНИЕ МОДЕЛИ (2020-2026)              ║")
    print("║  Модель этих данных НЕ видела при обучении!     ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # Загружаем данные
    df = pd.read_csv(INPUT_FILE)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Year'] = df['DateTime'].dt.year
    
    # Только тестовые данные (2020+)
    test = df[df['Year'] >= 2020].copy()
    print(f"\n📦 Тестовых событий: {len(test)}")
    
    # Фичи
    test['Month'] = test['DateTime'].dt.month
    test['Day'] = test['DateTime'].dt.day
    test['Hour'] = test['DateTime'].dt.hour
    test['DayOfWeek'] = test['DateTime'].dt.dayofweek
    test['Unit_Type'] = test['Event'].apply(determine_unit_type)
    test['Forecast'] = test['Forecast'].fillna(0)
    test['Previous'] = test['Previous'].fillna(0)
    test['Previous_Log'] = np.sign(test['Previous']) * np.log10(np.abs(test['Previous']) + 1)
    test['Forecast_Log'] = np.sign(test['Forecast']) * np.log10(np.abs(test['Forecast']) + 1)
    
    features = ['Currency', 'Event', 'Importance', 'Forecast', 'Previous',
                'Year', 'Month', 'Day', 'Hour', 'DayOfWeek',
                'Previous_Log', 'Forecast_Log']
    
    # Загружаем модели
    models = {}
    for unit in ['percent', 'index', 'jobs', 'currency']:
        path = f"{MODELS_DIR}/catboost_{unit}.cbm"
        if os.path.exists(path):
            model = CatBoostRegressor()
            model.load_model(path)
            models[unit.capitalize()] = model
            print(f"   ✅ Загружена: {unit}")
    
    if not models:
        print("❌ Модели не найдены! Сначала: python3 step4_train_model.py")
        return
    
    # ═══════════════════════════════════════════
    # ПРЕДСКАЗЫВАЕМ
    # 
    # Для каждой новости:
    # 1. Определяем тип (Percent/Index/Jobs/Currency)
    # 2. Загружаем нужную модель
    # 3. Предсказываем Факт
    # 4. Сравниваем с реальным Фактом
    # ═══════════════════════════════════════════
    print("\n🔮 Предсказываю...")
    
    predictions = []
    confidences = []
    
    for idx, row in test.iterrows():
        unit = row['Unit_Type']
        model = models.get(unit)
        
        if not model:
            predictions.append(np.nan)
            confidences.append(0)
            continue
        
        feat = [row[f] for f in features]
        
        try:
            raw = model.predict([feat])
            if len(raw.shape) > 1 and raw.shape[1] >= 2:
                pred = raw[0, 0]
                variance = max(raw[0, 1], 0)
                std = np.sqrt(variance)
            else:
                pred = raw[0] if len(raw.shape) == 1 else raw[0, 0]
                std = 0
            
            predictions.append(pred)
            
            # Уверенность: чем меньше std, тем увереннее
            scale = max(abs(row['Previous']), abs(row['Forecast']), 0.01)
            relative_std = std / scale
            conf = max(5, min(95, 95 - relative_std * 60))
            confidences.append(round(conf, 1))
        except:
            predictions.append(np.nan)
            confidences.append(0)
    
    test['Predicted'] = predictions
    test['Confidence'] = confidences
    
    # ═══════════════════════════════════════════
    # ОЦЕНИВАЕМ: Правильно или неправильно?
    #
    # Правильно = модель угадала НАПРАВЛЕНИЕ:
    #   Если Факт > Прогноз и Модель > Прогноз → ✅
    #   Если Факт < Прогноз и Модель < Прогноз → ✅
    #   Иначе → ❌
    # ═══════════════════════════════════════════
    
    eps = 1e-5
    test['Real_Direction'] = np.where(test['Actual'] > test['Forecast'] + eps, '📈 Выше', 
                             np.where(test['Actual'] < test['Forecast'] - eps, '📉 Ниже', '➡️ Равно'))
    test['Pred_Direction'] = np.where(test['Predicted'] > test['Forecast'] + eps, '📈 Выше',
                             np.where(test['Predicted'] < test['Forecast'] - eps, '📉 Ниже', '➡️ Равно'))
    test['Correct'] = test['Real_Direction'] == test['Pred_Direction']
    
    # Сохраняем результаты
    save_cols = ['DateTime', 'Event', 'Unit_Type', 'Forecast', 'Actual', 
                 'Predicted', 'Confidence', 'Real_Direction', 'Pred_Direction', 'Correct']
    test[save_cols].to_csv(OUTPUT_FILE, index=False)
    
    # ═══════════════════════════════════════════
    # ОТЧЁТ
    # ═══════════════════════════════════════════
    total = len(test)
    correct = test['Correct'].sum()
    accuracy = correct / total * 100
    
    print(f"""
╔══════════════════════════════════════════════════╗
║  📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ                     ║
╚══════════════════════════════════════════════════╝

  🎯 Общая точность: {accuracy:.1f}% ({correct}/{total})
  📊 Средняя уверенность: {test['Confidence'].mean():.1f}%
""")
    
    # По типам
    for unit in models.keys():
        udf = test[test['Unit_Type'] == unit]
        if len(udf) > 0:
            acc = udf['Correct'].sum() / len(udf) * 100
            conf = udf['Confidence'].mean()
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  🔹 {unit:10}: {acc:5.1f}% [{bar}] (уверенность: {conf:.0f}%)")
    
    # Примеры ✅
    print(f"\n  🏆 Примеры ПРАВИЛЬНЫХ предсказаний:")
    wins = test[test['Correct'] == True].head(3)
    for _, r in wins.iterrows():
        print(f"     ✅ {r['Event'][:35]:35} | Прогноз: {r['Forecast']:>8} | Факт: {r['Actual']:>8} | Модель: {r['Predicted']:>8.2f}")
    
    # Примеры ❌
    print(f"\n  💔 Примеры ОШИБОК:")
    losses = test[test['Correct'] == False].head(3)
    for _, r in losses.iterrows():
        print(f"     ❌ {r['Event'][:35]:35} | Прогноз: {r['Forecast']:>8} | Факт: {r['Actual']:>8} | Модель: {r['Predicted']:>8.2f}")
    
    print(f"""
  📁 Полный отчёт: {OUTPUT_FILE}

  Следующий шаг: python3 step6_predict_bot.py
""")


if __name__ == "__main__":
    test_model()
