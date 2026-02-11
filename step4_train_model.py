#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 4: ОБУЧЕНИЕ 4-х МОДЕЛЕЙ CatBoost КЛАССИФИКАТОР   ║
║  ЗАДАЧА: Выше или Ниже прогноза? (не точное число!)    ║
║  + Важность фичей + Автоподбор лучших                  ║
╚══════════════════════════════════════════════════════════╝

Запуск: python3 step4_train_model.py

СТАРАЯ ЗАДАЧА (регрессия):
  📊 "Факт будет 0.47%"  →  и потом сами сравниваем с прогнозом

НОВАЯ ЗАДАЧА (классификация):
  📈 "Выше прогноза" (73% уверенность)
  📉 "Ниже прогноза" (61% уверенность)
  
  Уверенность = вероятность из модели — не руками считаем!
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report
import os, json

# ═══════════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════════
INPUT_FILE = "data/enriched_calendar.csv"
MODELS_DIR = "models"
STATS_FILE = "models/event_stats.json"
FEATURES_FILE = "models/features_list.json"
IMPORTANCE_FILE = "models/feature_importance.json"
TRAIN_END = 2020
TEST_START = 2020


def train_all_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  🧠 КЛАССИФИКАТОР: Выше или Ниже прогноза?     ║")
    print("║  CatBoostClassifier + Feature Importance        ║")
    print("║  Обучение: 2007-2019 → Тест: 2020-2026          ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # --- Загрузка данных ---
    df = pd.read_csv(INPUT_FILE)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Базовые фичи
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Forecast'] = df['Forecast'].fillna(0)
    df['Previous'] = df['Previous'].fillna(0)
    df['Previous_Log'] = np.sign(df['Previous']) * np.log10(np.abs(df['Previous']) + 1)
    df['Forecast_Log'] = np.sign(df['Forecast']) * np.log10(np.abs(df['Forecast']) + 1)
    df['Forecast_Prev_Diff'] = df['Forecast'] - df['Previous']
    
    # ═══════════════════════════════════════════════════
    # ЦЕЛЕВАЯ ПЕРЕМЕННАЯ: 1 = Выше прогноза, 0 = Ниже
    # ═══════════════════════════════════════════════════
    # Убираем строки где Факт = Прогноз (нет направления)
    df['Direction'] = np.where(df['Actual'] > df['Forecast'], 1, 0)
    exact_match = (df['Actual'] == df['Forecast']).sum()
    print(f"\n📊 Факт = Прогноз (убираем): {exact_match} строк")
    df = df[df['Actual'] != df['Forecast']].reset_index(drop=True)
    
    # Фичи
    exclude_cols = ['DateTime', 'Actual', 'Unit_Type', 'Impact', 'Detail', 
                    'Actual_Raw', 'Forecast_Raw', 'Previous_Raw', 'Direction']
    features = [c for c in df.columns if c not in exclude_cols]
    cat_features = ['Currency', 'Event']
    
    # Заполняем NaN
    for f in features:
        if f not in cat_features and df[f].isna().any():
            df[f] = df[f].fillna(0)
    
    # Сохраняем список фичей
    with open(FEATURES_FILE, 'w') as f:
        json.dump(features, f, indent=2)
    
    total_features = len(features)
    ta_features = len([f for f in features if any(x in f for x in 
                  ['RSI','MACD','BB_','SMA','Trend','Swing','Wave','Vol_','OBV','MFI','VWAP','Consec'])])
    
    print(f"\n📦 Всего данных: {len(df)} (без точных совпадений)")
    print(f"📊 Фичей: {total_features} (из них {ta_features} тех.анализ)")
    print(f"📈 Баланс классов: Выше={df['Direction'].mean()*100:.1f}% / Ниже={100 - df['Direction'].mean()*100:.1f}%")
    
    event_stats = {}
    all_importances = {}
    
    # --- Обучаем 4 КЛАССИФИКАТОРА ---
    unit_types = ['Percent', 'Index', 'Jobs', 'Currency']
    
    for unit in unit_types:
        group = df[df['Unit_Type'] == unit]
        train = group[group['Year'] < TRAIN_END]
        test = group[group['Year'] >= TEST_START]
        
        if len(train) < 30:
            print(f"\n⚠️ {unit}: мало данных ({len(train)}). Пропускаю.")
            continue
        
        print(f"\n{'='*55}")
        print(f"🔹 {unit} Classifier")
        print(f"{'='*55}")
        print(f"   📚 Обучение: {len(train)} событий (2007-2019)")
        print(f"   📝 Тест:     {len(test)} событий (2020-2026)")
        print(f"   📈 Баланс: Выше={train['Direction'].mean()*100:.1f}% / Ниже={100 - train['Direction'].mean()*100:.1f}%")
        
        X_train, y_train = train[features], train['Direction']
        X_test, y_test = test[features], test['Direction']
        
        # ═══════════════════════════════════════════
        # CatBoostClassifier — предсказывает вероятность!
        # ═══════════════════════════════════════════
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            loss_function='Logloss',         # бинарная классификация
            eval_metric='AUC',               # Area Under Curve — лучшая метрика
            cat_features=cat_features,
            verbose=500,
            random_seed=42,
            l2_leaf_reg=5,                   # сильная регуляризация
            min_data_in_leaf=20,             # минимум данных в листе
            auto_class_weights='Balanced',   # балансировка классов
        )
        
        print(f"\n   ⏳ Обучение классификатора...")
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=200)
        
        # Сохраняем
        path = f"{MODELS_DIR}/catboost_{unit.lower()}.cbm"
        model.save_model(path)
        print(f"   💾 Сохранено: {path}")
        
        # ═══════════════════════════════════════════
        # ОЦЕНКА: Точность + Уверенность
        # ═══════════════════════════════════════════
        proba = model.predict_proba(X_test)        # вероятности [P(ниже), P(выше)]
        preds = model.predict(X_test)               # 0 или 1
        
        accuracy = accuracy_score(y_test, preds) * 100
        
        # Средняя уверенность
        confidence = proba.max(axis=1)  # берём max вероятность
        avg_confidence = confidence.mean() * 100
        
        # Уверенность когда модель ПРАВА
        correct_mask = (preds == y_test.values)
        conf_when_right = confidence[correct_mask].mean() * 100
        conf_when_wrong = confidence[~correct_mask].mean() * 100
        
        # Высокая уверенность (>60%)
        high_conf_mask = confidence > 0.60
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(y_test[high_conf_mask], preds[high_conf_mask]) * 100
            high_conf_count = high_conf_mask.sum()
        else:
            high_conf_accuracy = 0
            high_conf_count = 0
        
        print(f"\n   ╔════════════════════════════════════════════╗")
        print(f"   ║  📊 РЕЗУЛЬТАТЫ: {unit:20s}        ║")
        print(f"   ╠════════════════════════════════════════════╣")
        print(f"   ║  🎯 Точность:          {accuracy:5.1f}%              ║")
        print(f"   ║  📊 Сред. уверенность: {avg_confidence:5.1f}%              ║")
        print(f"   ║  ✅ Уверен. когда прав: {conf_when_right:5.1f}%              ║")
        print(f"   ║  ❌ Уверен. когда неправ:{conf_when_wrong:5.1f}%              ║")
        print(f"   ║  🔥 При уверен. >60%:   {high_conf_accuracy:5.1f}% ({high_conf_count} из {len(test)}) ║")
        print(f"   ╚════════════════════════════════════════════╝")
        
        # ═══════════════════════════════════════════
        # ВАЖНОСТЬ ФИЧЕЙ — какие реально помогают?
        # ═══════════════════════════════════════════
        importance = model.get_feature_importance()
        feat_imp = sorted(zip(features, importance), key=lambda x: -x[1])
        
        print(f"\n   🏆 Топ-10 важных фичей:")
        for i, (f_name, imp) in enumerate(feat_imp[:10]):
            bar = '█' * int(imp / max(importance) * 20)
            print(f"      {i+1:2d}. {f_name:30s} {imp:6.1f} {bar}")
        
        # Сохраняем важность
        all_importances[unit] = {name: float(imp) for name, imp in feat_imp}
        
        # Статистика по событиям
        for event_name in train['Event'].unique():
            mask_train = train['Event'].values == event_name
            mask_test = test['Event'].values == event_name
            if mask_train.sum() >= 3:
                if mask_test.sum() > 0:
                    event_acc = accuracy_score(y_test[mask_test], preds[mask_test]) * 100
                    event_conf = confidence[mask_test].mean() * 100
                else:
                    event_acc = 0
                    event_conf = 50
                event_stats[event_name] = {
                    'accuracy': float(event_acc),
                    'avg_confidence': float(event_conf),
                    'count': int(mask_train.sum()),
                    'unit_type': unit
                }
    
    # Сохраняем статистику
    with open(STATS_FILE, 'w') as f:
        json.dump(event_stats, f, indent=2)
    print(f"\n💾 Статистика событий: {STATS_FILE} ({len(event_stats)} событий)")
    
    with open(IMPORTANCE_FILE, 'w') as f:
        json.dump(all_importances, f, indent=2)
    print(f"💾 Важность фичей: {IMPORTANCE_FILE}")
    
    print(f"""
╔══════════════════════════════════════════════════╗
║  ✅ ВСЕ 4 КЛАССИФИКАТОРА ОБУЧЕНЫ!               ║
║                                                  ║
║  Теперь бот предсказывает:                       ║
║  📈 "Выше прогноза" (73% уверенность)           ║
║  📉 "Ниже прогноза" (61% уверенность)           ║
╚══════════════════════════════════════════════════╝

  📁 Модели в: {MODELS_DIR}/
  
  Следующий шаг: python3 step5_test_model.py
""")


if __name__ == "__main__":
    train_all_models()
