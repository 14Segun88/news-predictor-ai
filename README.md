# 🧠 News Predictor AI (Малыш v4.0)

Это AI-проект для предсказания реакции рынка (Forex/Stocks) на экономические новости (CPI, NFP, GDP и т.д.).
Использует гибридную нейросеть **PyTorch Fusion Network**, которая объединяет:
1.  📊 **Табличные данные** (139 фичей: RSI, VIX, DXY, разница прогнозов)
2.  📚 **Правила из книг** (Embeddings из Investopedia/BabyPips)
3.  🗣️ **Сентимент трейдеров** (Анализ "настроения толпы")

---

## 🚀 Как запустить (Quick Start)

Этот проект настроен для запуска "как есть" (as is). Вам нужен только Python 3.10+.

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
python3 -m playwright install chromium
```

### 2. Запуск бота
```bash
# Нужно задать токен вашего бота (получить у @BotFather)
export BOT_TOKEN='ваш_токен_здесь'

# Запуск
python3 step11_bot_fusion.py
```

---

## 📂 Структура проекта (По шагам)

Проект построен как обучающий курс. Вы можете пройти все шаги заново:

*   **Шаг 1:** `python3 step1_explain.py` — Теория ML (просто текст).
*   **Шаг 2:** `python3 step2_collect_data.py` — Сбор данных с investing.com (Playwright).
*   **Шаг 3:** `python3 step3_clean_data.py` — Очистка данных.
*   **Шаг 3b:** `python3 step3b_enrich_data.py` — Обогащение рыночными данными (yfinance).
*   **Шаг 4-5:** Обучение базового CatBoost (устарело, но работает).
*   **Шаг 7-8:** `python3 step7_prepare_books.py` + `step8_create_embeddings.py` — Создание базы знаний и Embeddings.
*   **Шаг 9:** `python3 step9_train_fusion.py` — **Обучение главной нейросети (Fusion Network)**.
*   **Шаг 11:** `python3 step11_bot_fusion.py` — **Финальный Telegram бот**.

---

## 🤖 Как пользоваться ботом

Отправьте боту данные в таком формате:

```text
Валюта-USD
Событие-CPI (м/м)
Важн.-3 звезды
Прогноз-0,4%
Пред.-0,6%
```

Бот проанализирует историю, применит правила и выдаст прогноз (ВЫШЕ/НИЖЕ) с уверенностью в %.

---

## 🛠 Требования
*   Python 3.10+
*   RAM: 4GB+ (для загрузки embeddings)
*   GPU: не обязательно (работает на CPU)
