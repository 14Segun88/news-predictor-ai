#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 7b: ГЕНЕРАЦИЯ ДИАЛОГОВ ТРЕЙДЕРОВ (SENTIMENT)      ║
║  Эмуляция чатов/форумов на основе Тех. Анализа         ║
║  (Чтобы модель училась "слышать" толпу)                ║
╚══════════════════════════════════════════════════════════╝

Мы создаём "Исторический сентимент" для обучения.
В реальности этот скрипт можно заменить на парсер Telegram/Investing.com.

Логика генерации (на основе Технических индикаторов):
- RSI > 70 → "Перекуплен", "Надо шортить", "Жду отката"
- RSI < 30 → "Дно", "Закупаем", "Отскок неизбежен"
- VIX > 25 → "Паника", "Страшно", "Кэш - король"
- Bull Trend → "To the moon", "Только лонг", "Выкупай просадки"
- Bear Trend → "Льём", "Это крах", "Ниже только дно"
"""

import pandas as pd
import numpy as np
import json
import random
import os

INPUT_FILE = "data/enriched_calendar.csv"
OUTPUT_FILE = "data/trader_chats.json"

# База фраз трейдеров
PHRASES = {
    "overbought": [
        "RSI зашкаливает, пора шортить!",
        "Рынок перегрет, жду коррекцию.",
        "Зафиксировал прибыль, слишком высоко забрались.",
        "Не верю в рост, дивергенция на часовике.",
        "Ну куда еще выше? Шорт с текущих.",
        "This is a bubble, selling everything.",
        "Overextended, looking for puts.",
        "Цена оторвалась от скользящей, будет возврат.",
        "Все в лонгах, значит скоро побреют.",
        "Продал, хватит жадничать."
    ],
    "oversold": [
        "RSI на дне, надо брать!",
        "Купил на всю котлету, это дно.",
        "Отскок неизбежен, перепроданность дикая.",
        "Скидки! Закупаемся глупцы.",
        "Ниже уже некуда, поддержка железобетонная.",
        "Buy the dip!",
        "Oversold conditions, bouncing soon.",
        "Страх на максимуме, время покупать.",
        "Институционалы тарят, я тоже.",
        "Отличная цена для входа в лонг."
    ],
    "bull_trend": [
        "Тренд наш друг, только лонг.",
        "To the moon! 🚀",
        "Пробил сопротивление, летим дальше.",
        "Любой откат выкупается, быки сильны.",
        "Доллар машина, прёт как танк.",
        "Strong uptrend, holding calls.",
        "Don't fight the trend.",
        "Вижу цель выше, держу позицию.",
        "Объёмы на рост есть, всё четко.",
        "Золотой крест, среднесрок вверх."
    ],
    "bear_trend": [
        "Тренд нисходящий, не ловите ножи.",
        "Льём! 📉",
        "Пробили поддержку, летим в ад.",
        "Ралли дохлой кошки, продавай на отскоке.",
        "Медведи давят, быков нет.",
        "Falling knife, stay away.",
        "Sell the rally.",
        "Слабость на лицо, идем ниже.",
        "Объёмы на селл огромные.",
        "Смертельный крест, всё плохо."
    ],
    "panic": [
        "Паника на рынке! 😱",
        "VIX улетел, закрываю всё.",
        "Кэш - король, сижу на заборе.",
        "Волатильность бешеная, вынесут всех.",
        "Лучше не лезть, мясорубка.",
        "Fear is extreme.",
        "Markets are crazy today.",
        "Маржин-коллы летят, осторожнее.",
        "Ничего не понятно, выключил терминал.",
        "Жесть что творится."
    ],
    "calm": [
        "Боковик, скука...",
        "Ничего не происходит, ждем новости.",
        "Флэт, торгую от границ.",
        "На рынке тишина.",
        "Ждем импульс, пока спим.",
        "Choppy market.",
        "Sideways action.",
        "Копим силы перед рывком.",
        "Объёмов нет, рынок мёртв.",
        "Консолидация."
    ]
}

def generate_sentiment():
    print("🗣️ ГЕНЕРАЦИЯ ДИАЛОГОВ ТРЕЙДЕРОВ...")
    df = pd.read_csv(INPUT_FILE)
    
    # Предполагаем, что у нас есть RSI, VIX, SMA из enrich_data
    # Если их нет в явном виде, сгенерируем заглушки для демонстрации
    # В реальном проекте используем реальные колонки
    
    # Проверка колонок
    available_cols = df.columns.tolist()
    print(f"📊 Доступные колонки: {len(available_cols)}")
    
    chat_data = {}
    
    for i, row in df.iterrows():
        event = row['Event']
        
        # Симуляция тех. показателей (если их нет, берем рандом с bias от Actual-Forecast)
        # В идеале тут должны быть реальные значения RSI_14, VIX_Close и т.д.
        # Для демонстрации Fusion Model мы СИМУЛИРУЕМ поведение трейдеров,
        # которое ОБЫЧНО коррелирует с рынком.
        
        # Логика "Толпы":
        # Толпа часто ошибается, но создает тренд.
        # Мы будем генерировать микс мнений.
        
        messages = []
        
        # 1. RSI Logic (Симуляция)
        # В реальности берем row['RSI_14']
        rsi = random.gauss(50, 15) 
        
        if rsi > 70:
            messages.extend(random.sample(PHRASES["overbought"], k=2))
        elif rsi < 30:
            messages.extend(random.sample(PHRASES["oversold"], k=2))
            
        # 2. Trend Logic
        # Симуляция тренда
        trend = random.choice(['bull', 'bear', 'flat'])
        if trend == 'bull':
            messages.extend(random.sample(PHRASES["bull_trend"], k=2))
        elif trend == 'bear':
            messages.extend(random.sample(PHRASES["bear_trend"], k=2))
        else:
            messages.extend(random.sample(PHRASES["calm"], k=1))
            
        # 3. Panic Logic
        if random.random() < 0.1: # 10% шанс паники
            messages.extend(random.sample(PHRASES["panic"], k=1))
            
        # Перемешиваем
        random.shuffle(messages)
        
        # Объединяем в один текст "обсуждения"
        chat_text = " | ".join(messages)
        chat_data[event] = chat_text # Маппинг по событию (упрощенно)
        
        # Важно: нам нужно мапить не только по имени события, но и по времени, 
        # но для PoC привяжем к типу события или индексу
        
    # Создадим список списков, соответствующий строкам DF
    # Чтобы step8 мог взять row-by-row
    
    all_chats = []
    print(f"⏳ Генерация сообщений для {len(df)} событий...")
    
    for i, row in df.iterrows():
        # Генерируем уникальный сентимент для КАЖДОЙ строки (даже если события одинаковые, время разное)
        # В реальности тут был бы парсинг чата за дату row['DateTime']
        
        msgs = []
        
        # Используем реальные данные если есть, иначе рандом
        # (Предположим column 'Previous' как прокси тренда для примера)
        
        prev = row.get('Previous', 0)
        fc = row.get('Forecast', 0)
        
        if fc > prev:
            msgs.extend(random.sample(PHRASES["bull_trend"], k=1))
        elif fc < prev:
            msgs.extend(random.sample(PHRASES["bear_trend"], k=1))
            
        # Добавляем рандомного шума (форумы зашумлены)
        msgs.extend(random.sample(PHRASES["calm"], k=1))
        
        all_chats.append(" ".join(msgs))
        
    # Сохраняем как JSON список строк
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chats, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Успешно сгенерировано {len(all_chats)} диалогов.")
    print(f"📁 Сохранено в {OUTPUT_FILE}")
    print("Пример:")
    print(all_chats[0])

if __name__ == "__main__":
    generate_sentiment()
