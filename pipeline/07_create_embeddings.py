#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 8: СОЗДАНИЕ ЭМБЕДДИНГОВ ИЗ БАЗЫ ЗНАНИЙ           ║
║  sentence-transformers → 384-dim vectors                ║
║  Для каждого события: подбираем релевантные правила     ║
╚══════════════════════════════════════════════════════════╝
"""
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import os

KNOWLEDGE_FILE = "data/knowledge_base.json"
ENRICHED_FILE = "data/enriched_calendar.csv"
OUTPUT_FILE = "data/event_embeddings.npy"
MAPPING_FILE = "data/event_rule_mapping.json"

# ═══════════════════════════════════════════════════
# Для каждого типа события подбираем правила
# ═══════════════════════════════════════════════════
def match_rules_to_event(event_name, rules):
    """Находим релевантные правила для данного события."""
    matched = []
    event_lower = event_name.lower()
    
    for rule in rules:
        events = rule.get("events", [])
        # Wildcard * = подходит для всех
        if "*" in events:
            matched.append(rule)
            continue
        # Проверяем совпадение по имени
        for ev in events:
            if ev.lower() in event_lower or event_lower in ev.lower():
                matched.append(rule)
                break
    
    return matched


def create_embeddings():
    print("╔══════════════════════════════════════════════════╗")
    print("║  🧠 СОЗДАНИЕ ЭМБЕДДИНГОВ (sentence-transformers) ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # Загрузка модели
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Устройство: {device}")
    print(f"📦 Загрузка модели all-MiniLM-L6-v2...")
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    print(f"   ✅ Модель загружена ({device})")
    
    # Загрузка правил
    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    print(f"📚 Правил в базе знаний: {len(rules)}")
    
    # Кодируем ВСЕ правила в embeddings
    print(f"\n⏳ Кодирование правил в embeddings...")
    rule_texts = [r['text'] for r in rules]
    rule_embeddings = model.encode(rule_texts, show_progress_bar=True, batch_size=32)
    print(f"   ✅ {len(rule_embeddings)} правил → {rule_embeddings.shape[1]}-dim vectors")
    
    # Загрузка датасета
    df = pd.read_csv(ENRICHED_FILE)
    print(f"\n📊 Событий в датасете: {len(df)}")
    
    # Для каждого УНИКАЛЬНОГО события создаём composite embedding
    unique_events = df['Event'].unique()
    print(f"📋 Уникальных типов событий: {len(unique_events)}")
    
    event_to_embedding = {}
    event_to_rules = {}
    
    for event_name in unique_events:
        matched = match_rules_to_event(event_name, rules)
        
        if matched:
            # Берём тексты подходящих правил
            matched_texts = [r['text'] for r in matched]
            matched_embs = model.encode(matched_texts, batch_size=32)
            # Средний embedding всех релевантных правил
            composite = matched_embs.mean(axis=0)
        else:
            # Нет подходящих правил — создаём embedding из самого названия
            composite = model.encode([f"Economic indicator: {event_name}"])[0]
        
        event_to_embedding[event_name] = composite
        event_to_rules[event_name] = [r.get('text_ru', r['text'])[:80] for r in matched[:5]]
    
    # Создаём embedding для КАЖДОЙ строки датасета
    print(f"\n⏳ Создание RULE embeddings для {len(df)} строк...")
    rule_embeddings_out = np.zeros((len(df), rule_embeddings.shape[1]), dtype=np.float32)
    
    for i, row in df.iterrows():
        event_name = row['Event']
        rule_embeddings_out[i] = event_to_embedding.get(event_name, np.zeros(rule_embeddings.shape[1]))
    
    np.save("data/event_embeddings.npy", rule_embeddings_out)
    
    # --- SENTIMENT EMBEDDINGS ---
    print(f"\n⏳ Создание SENTIMENT embeddings (из чатов)...")
    try:
        with open("data/trader_chats.json", 'r', encoding='utf-8') as f:
            chats = json.load(f)
        
        if len(chats) != len(df):
            print(f"⚠️ Внимание: {len(chats)} диалогов != {len(df)} строк. Обрезаем/дополняем.")
            chats = chats[:len(df)] + [""] * (len(df) - len(chats))
            
        sentiment_embeddings = model.encode(chats, show_progress_bar=True, batch_size=64)
        np.save("data/sentiment_embeddings.npy", sentiment_embeddings)
        print(f"   ✅ Sentiment embeddings: {sentiment_embeddings.shape}")
        
    except FileNotFoundError:
        print("⚠️ Файл data/trader_chats.json не найден. Пропуск sentiment embeddings.")
    # ---------------------------

    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(event_to_rules, f, ensure_ascii=False, indent=2)
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ ЭМБЕДДИНГИ СОЗДАНЫ!                                ║
╚══════════════════════════════════════════════════════════╝

  📁 Rules Embeddings: data/event_embeddings.npy
  📁 Sentiment Embeddings: data/sentiment_embeddings.npy (если есть)
  
  📁 Маппинг: {MAPPING_FILE}
     Событий с подходящими правилами: {sum(1 for v in event_to_rules.values() if v)}
  
  Примеры маппинга:
""")
    for event, matched_rules in list(event_to_rules.items())[:5]:
        print(f"  📰 {event}:")
        for r in matched_rules[:2]:
            print(f"     📚 {r}")
        print()
    
    print(f"  Следующий шаг: python3 step9_train_fusion.py")


if __name__ == "__main__":
    create_embeddings()
