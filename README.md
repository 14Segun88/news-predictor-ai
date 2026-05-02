# News Predictor AI — Hybrid ML System for Financial Event Prediction

> **Problem:** Трейдеры и аналитики реагируют на экономические новости (CPI, NFP, GDP) интуитивно, без систематического анализа исторических паттернов. Ручной анализ влияния макроэкономических событий на валютные пары (EUR, USD, RUB) занимает часы.
>
> **Solution:** Гибридная ML-система, объединяющая 3 источника сигналов: табличные данные (139 фичей), knowledge base из Investopedia/BabyPips (embeddings), и анализ трейдерского сентимента — через PyTorch Fusion Network + 4 специализированных CatBoost-модели.
>
> **Outcome:** 54.16% overall accuracy на 7865 исторических событиях (baseline 50% на эффективном рынке). При confidence >65% — 85.7% accuracy. Telegram-бот выдаёт прогноз за секунды.

### Key Metrics
| Metric | Value |
|---|---|
| Обучающая выборка | 7 865 экономических событий |
| Валюты | EUR, USD, RUB |
| Табличных фичей | 139 (RSI, VIX, DXY, delta прогнозов) |
| CatBoost-моделей | 4 (currency, percent, jobs, index) |
| Overall accuracy | 54.16% (random baseline = 50%) |
| High-confidence accuracy (>65%) | 85.7% |
| ML-приёмы | LayerNorm, AdamW, Cosine Annealing, gradient clipping, time-based split |

### Pipeline Architecture

```mermaid
flowchart LR
    subgraph Data Collection
        A[investing.com<br/>Playwright] --> B[Raw Calendar]
        C[yfinance] --> D[Market Data<br/>RSI, VIX, DXY]
    end
    
    subgraph Feature Engineering
        B --> E[Clean & Enrich<br/>139 features]
        D --> E
    end
    
    subgraph Knowledge Base
        F[Investopedia<br/>BabyPips] --> G[Book Embeddings<br/>sentence-transformers]
        H[Trader Chats] --> I[Sentiment Embeddings]
    end
    
    subgraph Models
        E --> J[4× CatBoost<br/>currency/percent/jobs/index]
        E --> K[PyTorch Fusion Network]
        G --> K
        I --> K
        J --> K
    end
    
    K --> L[🤖 Telegram Bot<br/>UP/DOWN + confidence %]
```

### Tech Stack
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat)
![Playwright](https://img.shields.io/badge/Playwright-2EAD33?style=flat&logo=playwright)
![aiogram](https://img.shields.io/badge/aiogram-2CA5E0?style=flat&logo=telegram)

### Demo
🎬 [Видео-демо работы бота](#)

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
# Скопируйте .env.example в .env и укажите токен:
cp .env.example .env

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
