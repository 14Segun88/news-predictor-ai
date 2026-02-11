#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 10: МОЗГ БОТА "MALISH" (3-Branch Fusion)          ║
║  CatBoost + Books + Trader Sentiment → Verdict          ║
╚══════════════════════════════════════════════════════════╝

Пример запуска:
python3 step10_final_prediction.py --event "CPI" --currency "USD" --forecast 0.4 --previous 0.6 --importance 3
"""
import argparse
import pandas as pd
import numpy as np
import torch
import json
import os
import random
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from step9_train_fusion import FusionNetwork  # Импортируем 3-branch архитектуру

# ═══════════════════════════════════════════════════
# НАСТРОЙКИ
# ═══════════════════════════════════════════════════
MODELS_DIR = "models"
DATA_DIR = "data"
CATBOOST_FEATURES = "models/features_list.json"
FUSION_NORM = "models/fusion_norm_stats.json"
KNOWLEDGE_MAPPING = "data/event_rule_mapping.json"
MARKET_DATA = "data/market_data_v3.csv"   # В реальности тут нужен Live Data Source

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Фразы для генерации "Сентимента" (копия логики из step7b)
TRADER_PHRASES = {
    "bull": ["Тренд наш друг, только лонг.", "To the moon! 🚀", "Выкупай просадки."],
    "bear": ["Льём! 📉", "Пробили поддержку, летим в ад.", "Sell the rally."],
    "flat": ["Боковик, скукота...", "Ждем импульс."],
    "panic": ["Паника! VIX улетел.", "Страшно, лучше не лезть."],
    "overbought": ["RSI перегрет, шорт.", "Дивергенция, продавай."],
    "oversold": ["RSI на дне, тарь!", "Скидки, надо брать."]
}

def load_resources():
    print(f"⏳ Загрузка мозгов Malish ({DEVICE})...")
    
    with open(CATBOOST_FEATURES, 'r') as f:
        cb_features = json.load(f)
    
    # Market Data (последние исторические для демо)
    # В ПРОДЕ: Тут должен быть API запрос к yfinance/tradingview
    if os.path.exists(MARKET_DATA):
        market_df = pd.read_csv(MARKET_DATA)
        last_market = market_df.iloc[-1].to_dict()
    else:
        # Fallback defaults
        last_market = {'SP500_RSI': 50, 'VIX_Close': 15, 'DXY_Close': 100, 'DXY_SMA50': 99}
    
    with open(KNOWLEDGE_MAPPING, 'r') as f:
        event_rules = json.load(f)
    
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
    
    with open(FUSION_NORM, 'r') as f:
        norm_stats = json.load(f)
    
    # 3-Branch Model
    fusion_model = FusionNetwork(tabular_dim=len(norm_stats['features']), text_dim=384).to(DEVICE)
    checkpoint = torch.load(f"{MODELS_DIR}/fusion_model.pt", map_location=DEVICE, weights_only=True)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    fusion_model.eval()
    
    return cb_features, last_market, event_rules, embedder, norm_stats, fusion_model


def get_catboost_model(unit_type):
    model_path = f"{MODELS_DIR}/catboost_{unit_type.lower()}.cbm"
    if not os.path.exists(model_path): return None
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def determine_unit_type(event_name):
    ev = event_name.lower()
    if 'rate' in ev or 'cpi' in ev or 'gdp' in ev: return 'Percent'
    elif 'pmi' in ev or 'confidence' in ev: return 'Index'
    elif 'employment' in ev or 'job' in ev: return 'Jobs'
    return 'Percent'


def generate_trader_sentiment_text(last_market, forecast, previous):
    """Генерирует текст сентимента на основе текущего рынка (симуляция чата)."""
    msgs = []
    
    # Technical Logic
    rsi = last_market.get('SP500_RSI', 50)
    vix = last_market.get('VIX_Close', 15)
    
    if rsi > 70: msgs.extend(random.sample(TRADER_PHRASES["overbought"], 1))
    elif rsi < 30: msgs.extend(random.sample(TRADER_PHRASES["oversold"], 1))
    
    if vix > 25: msgs.extend(random.sample(TRADER_PHRASES["panic"], 1))
    
    # Fundamental Logic
    if forecast > previous: msgs.extend(random.sample(TRADER_PHRASES["bull"], 1))
    elif forecast < previous: msgs.extend(random.sample(TRADER_PHRASES["bear"], 1))
    else: msgs.extend(random.sample(TRADER_PHRASES["flat"], 1))
    
    return " ".join(msgs)


def predict(args):
    # 1. Init
    cb_features, last_market, event_rules, embedder, norm_stats, fusion_model = load_resources()
    
    # 2. Input Data
    unit_type = determine_unit_type(args.event)
    
    # Имитация сбора данных
    input_data = {f: 0.0 for f in cb_features}
    input_data['Forecast'] = args.forecast
    input_data['Previous'] = args.previous
    input_data['Importance'] = args.importance
    input_data['Forecast_Prev_Diff'] = args.forecast - args.previous
    
    # Заполняем рыночными
    for k, v in last_market.items():
        if k in input_data: input_data[k] = v
    input_data['Event'] = args.event
    input_data['Currency'] = args.currency
    
    df = pd.DataFrame([input_data])
    
    # 3. CatBoost Prediction
    cb_model = get_catboost_model(unit_type)
    if cb_model:
        cb_proba = cb_model.predict_proba(df)[0]
        cb_pred = cb_proba.argmax()
        cb_conf = cb_proba[cb_pred] * 100
        cb_dir = "ВЫШЕ 📈" if cb_pred == 1 else "НИЖЕ 📉"
    else:
        cb_dir, cb_conf = "N/A", 0
    
    # 4. Fusion Prediction (3 Branches)
    
    # A. Tabular
    num_features = norm_stats['features']
    tab_vector = [df.iloc[0].get(f, 0.0) for f in num_features]
    tab_vector = np.array(tab_vector, dtype=np.float32)
    mean = np.array(norm_stats['mean'], dtype=np.float32)
    std = np.array(norm_stats['std'], dtype=np.float32)
    tab_tensor = torch.FloatTensor((tab_vector - mean) / std).unsqueeze(0).to(DEVICE)
    
    # B. Rules
    rules_text = event_rules.get(args.event, [])
    
    # Fuzzy match fallback
    if not rules_text:
        # Пытаемся найти частичное совпадение
        candidates = [k for k in event_rules.keys() if args.event.lower() in k.lower()]
        if candidates:
            # Берём первый попавшийся или самый короткий (как самый общий)
            best_match = min(candidates, key=len)
            rules_text = event_rules[best_match]
            print(f"\n   ℹ️  (Правила взяты для '{best_match}')")

    if rules_text:
        rule_emb = embedder.encode(rules_text)
        rule_emb = np.mean(rule_emb, axis=0, keepdims=True)
        rule_desc = "\n".join([f"   • {r[:80]}..." for r in rules_text[:3]])
    else:
        rule_emb = embedder.encode([f"Economic indicator: {args.event}"])
        rule_desc = "   (Нет спец. правил)"
    rule_tensor = torch.FloatTensor(rule_emb).to(DEVICE)
    
    # C. Sentiment (Generated Live)
    sentiment_text = generate_trader_sentiment_text(last_market, args.forecast, args.previous)
    sent_emb = embedder.encode([sentiment_text])
    sent_tensor = torch.FloatTensor(sent_emb).to(DEVICE)
    
    # Forward
    with torch.no_grad():
        fusion_out = fusion_model(tab_tensor, rule_tensor, sent_tensor)
        fusion_probs = torch.softmax(fusion_out, dim=1).cpu().numpy()[0]
        fus_pred = fusion_probs.argmax()
    
    fus_conf = fusion_probs[fus_pred] * 100
    fus_dir = "ВЫШЕ 📈" if fus_pred == 1 else "НИЖЕ 📉"
    
    # 5. Output
    print("\n" + "="*60)
    print(f"🤖 ТОРГОВЫЙ СИГНАЛ: {args.event} ({args.currency})")
    print(f"   Прогноз: {args.forecast} | Пред: {args.previous}")
    print("="*60)
    
    print(f"\n1️⃣  CatBoost (Базовая модель):")
    print(f"   \"{cb_dir} прогноза (вероятность {cb_conf:.1f}%)\"")
    
    print(f"\n2️⃣  Fusion Model (Нейросеть + Книги + Чаты):")
    print(f"   \"Я согласна! Мои знания говорят:")
    print(f"\n   📈 Технический фон:")
    print(f"   • RSI: {last_market.get('SP500_RSI', 50):.0f} | VIX: {last_market.get('VIX_Close', 15):.1f}")
    if last_market.get('DXY_Close', 100) > last_market.get('DXY_SMA50', 99):
        print(f"   • Тренд доллара восходящий (DXY > SMA50).")
    
    print(f"\n   📚 Правила из книг:")
    print(rule_desc)
    
    print(f"\n   🗣️ Трейдеры в чатах:")
    print(f"   • '{sentiment_text}'")
    
    print(f"\n   🎯 Моя уверенность: {fus_conf:.1f}%\"")
    
    # Verdict
    print("\n" + "-"*60)
    print("🚦  ВЕРДИКТ БОТА:")
    
    if cb_pred == fus_pred:
        if fus_conf > 70:
            print(f"   STRONG BUY {args.currency} 🟢")
            print(f"   (Обе модели согласны, уверенность > 70%)")
        elif fus_conf > 60:
            print(f"   CONFIRMED TRADE 🟡")
        else:
            print(f"   WEAK SIGNAL ⚪")
    else:
        print(f"   NEUTRAL / CONFLICT 🔴")
        print(f"   (Модели дают разные сигналы)")
    print("-"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=str, required=True)
    parser.add_argument("--currency", type=str, default="USD")
    parser.add_argument("--forecast", type=float, required=True)
    parser.add_argument("--previous", type=float, required=True)
    parser.add_argument("--importance", type=int, default=3)
    args = parser.parse_args()
    predict(args)
