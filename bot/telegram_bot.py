#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 11: TELEGRAM BOT «МАЛЫШ» v4.0 (FUSION NETWORK)    ║
║                                                          ║
║  PyTorch Fusion (3 ветки):                               ║
║  1. CatBoost Tabular  — 139 фичей из enriched_calendar  ║
║  2. Rules (TA/FA)     — embeddings из event_embeddings   ║
║  3. Sentiment (чаты)  — embeddings из sentiment_emb      ║
║                                                          ║
║  При инференсе:                                          ║
║  → Находим похожие события в истории                     ║
║  → Берём их РЕАЛЬНЫЕ фичи и embeddings                   ║
║  → Подаём в обученную нейросеть                          ║
╚══════════════════════════════════════════════════════════╝
"""
import asyncio
import logging
import os
import re
import numpy as np
import pandas as pd
import torch
import json

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN  = os.getenv("BOT_TOKEN")
MODELS_DIR = "models"
DATA_DIR   = "data"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════
#  АРХИТЕКТУРА (Точная копия step9_train_fusion.py)
# ═══════════════════════════════════════════════════════
class FusionNetwork(torch.nn.Module):
    def __init__(self, tabular_dim, text_dim, hidden_dim=128):
        super().__init__()
        self.tabular_branch = torch.nn.Sequential(
            torch.nn.Linear(tabular_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
        )
        self.rules_branch = torch.nn.Sequential(
            torch.nn.Linear(text_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
        )
        self.sentiment_branch = torch.nn.Sequential(
            torch.nn.Linear(text_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
        )
        fusion_input = hidden_dim * 3
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_input, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, tabular, rules, sentiment):
        t = self.tabular_branch(tabular)
        r = self.rules_branch(rules)
        s = self.sentiment_branch(sentiment)
        return self.fusion(torch.cat([t, r, s], dim=1))


# ═══════════════════════════════════════════════════════
#  МОЗГ МАЛЫША
# ═══════════════════════════════════════════════════════
class MalishBrain:

    # Маппинг RU → EN
    EVENT_MAP = {
        "розничн":    "Retail Sales m/m",
        "retail":     "Retail Sales m/m",
        "cpi м/м":    "CPI m/m",
        "cpi г/г":    "CPI y/y",
        "cpi":        "CPI m/m",
        "нфп":        "Non-Farm Employment Change",
        "nfp":        "Non-Farm Employment Change",
        "non-farm":   "Non-Farm Employment Change",
        "ввп":        "GDP q/q",
        "gdp":        "GDP q/q",
        "pmi произв": "ISM Manufacturing PMI",
        "pmi":        "ISM Manufacturing PMI",
        "безработ":   "Unemployment Rate",
        "unemployment":"Unemployment Rate",
        "ставк":      "Federal Funds Rate",
        "rate":       "Federal Funds Rate",
        "нефт":       "Crude Oil Inventories",
        "crude oil":  "Crude Oil Inventories",
        "торгов":     "Trade Balance",
        "trade":      "Trade Balance",
        "ppi":        "PPI m/m",
        "потреб":     "CB Consumer Confidence",
        "consumer":   "CB Consumer Confidence",
        "заявки":     "Unemployment Claims",
        "claims":     "Unemployment Claims",
        "adp":        "ADP Non-Farm Employment Change",
        "средн почас":"Average Hourly Earnings m/m",
        "hourly":     "Average Hourly Earnings m/m",
        "зараб":      "Average Hourly Earnings m/m",
        "зарплат":    "Average Hourly Earnings m/m",
    }

    def __init__(self):
        self.history_df = None            # enriched_calendar.csv (30K строк)
        self.event_rules = {}             # event_rule_mapping.json
        self.trader_chats = []            # trader_chats.json
        self.event_embeddings = None      # (30102, 384) — ветка Rules
        self.sentiment_embeddings = None  # (30102, 384) — ветка Sentiment
        self.fusion_model = None          # PyTorch
        self.norm_stats = None            # mean/std для нормализации
        self.features_list = []           # 139 имён фичей

    def load(self):
        print("⏳ Загрузка мозгов Малыша...\n")

        # 1. История
        print("   📊 enriched_calendar.csv...")
        self.history_df = pd.read_csv(f"{DATA_DIR}/enriched_calendar.csv")
        print(f"      → {len(self.history_df)} строк")

        # 2. Правила из книг
        print("   📚 event_rule_mapping.json...")
        with open(f"{DATA_DIR}/event_rule_mapping.json", "r") as f:
            self.event_rules = json.load(f)
        print(f"      → {len(self.event_rules)} событий")

        # 3. Чаты
        print("   🗣️ trader_chats.json...")
        with open(f"{DATA_DIR}/trader_chats.json", "r") as f:
            self.trader_chats = json.load(f)
        print(f"      → {len(self.trader_chats)} чатов")

        # 4. Embeddings (pre-computed при обучении)
        print("   🧮 event_embeddings.npy + sentiment_embeddings.npy...")
        self.event_embeddings = np.load(f"{DATA_DIR}/event_embeddings.npy")
        self.sentiment_embeddings = np.load(f"{DATA_DIR}/sentiment_embeddings.npy")
        print(f"      → Rules: {self.event_embeddings.shape}")
        print(f"      → Sentiment: {self.sentiment_embeddings.shape}")

        # 5. Normalization stats
        print("   📐 fusion_norm_stats.json...")
        with open(f"{MODELS_DIR}/fusion_norm_stats.json", "r") as f:
            self.norm_stats = json.load(f)
        self.features_list = self.norm_stats["features"]
        print(f"      → {len(self.features_list)} фичей")

        # 6. Fusion Model
        print("   🧠 fusion_model.pt...")
        tab_dim = len(self.features_list)
        self.fusion_model = FusionNetwork(tabular_dim=tab_dim, text_dim=384).to(DEVICE)
        ckpt = torch.load(f"{MODELS_DIR}/fusion_model.pt", map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.fusion_model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.fusion_model.load_state_dict(ckpt)
        self.fusion_model.eval()
        print(f"      → Загружена (tabular_dim={tab_dim}, text_dim=384)")

        print("\n✅ Мозг Малыша загружен и готов!\n")

    # ─── Маппинг RU → EN ───
    def resolve_event(self, raw: str) -> str:
        low = raw.lower().strip()
        for key, en_name in self.EVENT_MAP.items():
            if key in low:
                return en_name
        return raw

    # ─── Поиск похожих событий в истории ───
    def find_similar_events(self, event_en: str, currency: str,
                            forecast: float, previous: float):
        """
        Находит в enriched_calendar.csv строки с таким же Event и Currency.
        Подбирает ближайшие по (Forecast - Previous).
        Возвращает индексы строк.
        """
        df = self.history_df
        mask = df["Event"].str.lower() == event_en.lower()
        if currency:
            mask &= df["Currency"] == currency
        matches = df[mask]

        if len(matches) == 0:
            # Fuzzy
            for part in event_en.lower().split():
                if len(part) > 3:
                    mask = df["Event"].str.lower().str.contains(part, na=False)
                    if currency:
                        mask &= df["Currency"] == currency
                    matches = df[mask]
                    if len(matches) > 0:
                        break

        if len(matches) == 0:
            return None, None, None

        # Берём ПОСЛЕДНЮЮ строку — она самая актуальная (самые свежие рыночные данные)
        # + подбираем ещё по условию Forecast vs Previous
        user_diff = forecast - previous

        matches = matches.copy()
        matches["_diff"] = matches["Forecast"] - matches["Previous"]
        matches["_dist"] = (matches["_diff"] - user_diff).abs()

        # Сортируем по близости условий, берём последние 10
        best = matches.nlargest(10, "Forecast")  # Последние по времени с похожими
        closest = matches.sort_values("_dist").head(10)  # Самые похожие по условиям

        # Объединяем
        selected = pd.concat([best, closest]).drop_duplicates()

        # Самая похожая строка — для табличного вектора
        best_row_idx = closest.index[0]

        # Статистика по направлению
        all_above = (matches["Actual"] > matches["Forecast"]).sum()
        all_total = len(matches)

        # Статистика по похожим условиям (тот же знак diff)
        if user_diff < 0:
            sim = matches[matches["_diff"] < 0]
        elif user_diff > 0:
            sim = matches[matches["_diff"] > 0]
        else:
            sim = matches[matches["_diff"].abs() < 0.01]

        if len(sim) > 0:
            sim_above = (sim["Actual"] > sim["Forecast"]).sum()
            sim_total = len(sim)
        else:
            sim_above = all_above
            sim_total = all_total
            sim = matches

        return {
            "best_row_idx": best_row_idx,
            "selected_indices": selected.index.tolist(),
            "event_name": matches["Event"].iloc[0],
            "total": all_total,
            "total_above": int(all_above),
            "sim_total": sim_total,
            "sim_above": int(sim_above),
            "pct_above": sim_above / sim_total * 100 if sim_total > 0 else 50,
            "avg_actual": sim["Actual"].mean(),
            "last5": matches.tail(5)[["Actual_Raw", "Forecast_Raw", "Previous_Raw"]].values.tolist(),
        }

    # ─── Построение табличного вектора ───
    def build_tabular_vector(self, best_row_idx: int,
                             forecast: float, previous: float,
                             importance: int) -> np.ndarray:
        """
        Берёт РЕАЛЬНЫЙ вектор из enriched_calendar.csv[best_row_idx]
        и подставляет пользовательские Forecast/Previous/Importance.
        """
        row = self.history_df.iloc[best_row_idx]
        vec = []
        for f in self.features_list:
            if f == "Forecast":
                vec.append(forecast)
            elif f == "Previous":
                vec.append(previous)
            elif f == "Importance":
                vec.append(float(importance))
            elif f == "Forecast_Prev_Diff":
                vec.append(forecast - previous)
            elif f in row.index:
                val = row[f]
                vec.append(float(val) if pd.notna(val) else 0.0)
            else:
                vec.append(0.0)
        return np.array(vec, dtype=np.float32)

    # ─── Нормализация ───
    def normalize(self, vec: np.ndarray) -> np.ndarray:
        mean = np.array(self.norm_stats["mean"], dtype=np.float32)
        std  = np.array(self.norm_stats["std"], dtype=np.float32)
        normed = (vec - mean) / std
        return np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    # ─── ГЛАВНЫЙ ПРОГНОЗ ───
    def predict(self, event_ru: str, currency: str, importance: int,
                forecast: float, previous: float) -> str:
        event_en = self.resolve_event(event_ru)

        # 1. Найти похожие события
        hist = self.find_similar_events(event_en, currency, forecast, previous)
        if hist is None:
            return (f"❌ Событие <b>{event_en}</b> не найдено в истории.\n"
                    f"Попробуй: CPI, NFP, GDP, Retail Sales, Unemployment Rate")

        best_idx = hist["best_row_idx"]
        sel_indices = hist["selected_indices"]

        # ═══ ВЕТКА 1: TABULAR (139 фичей из реальной строки) ═══
        tab_vec = self.build_tabular_vector(best_idx, forecast, previous, importance)
        tab_normed = self.normalize(tab_vec)
        tab_tensor = torch.FloatTensor(tab_normed).unsqueeze(0).to(DEVICE)

        # ═══ ВЕТКА 2: RULES (embeddings из event_embeddings.npy) ═══
        # Берём embeddings для похожих строк и усредняем
        valid_rule_idx = [i for i in sel_indices if i < len(self.event_embeddings)]
        if valid_rule_idx:
            rule_emb = self.event_embeddings[valid_rule_idx].mean(axis=0, keepdims=True)
        else:
            rule_emb = self.event_embeddings[best_idx:best_idx+1]
        rule_tensor = torch.FloatTensor(rule_emb).to(DEVICE)

        # ═══ ВЕТКА 3: SENTIMENT (embeddings из sentiment_embeddings.npy) ═══
        valid_sent_idx = [i for i in sel_indices if i < len(self.sentiment_embeddings)]
        if valid_sent_idx:
            sent_emb = self.sentiment_embeddings[valid_sent_idx].mean(axis=0, keepdims=True)
        else:
            sent_emb = self.sentiment_embeddings[best_idx:best_idx+1]
        sent_tensor = torch.FloatTensor(sent_emb).to(DEVICE)

        # ═══ FORWARD PASS ═══
        with torch.no_grad():
            out = self.fusion_model(tab_tensor, rule_tensor, sent_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())

        fus_conf = probs[pred] * 100
        fus_dir  = "ВЫШЕ 📈" if pred == 1 else "НИЖЕ 📉"

        # ═══ Правила из книг (для объяснения) ═══
        rules = self.event_rules.get(hist["event_name"], [])
        if not rules:
            for k in self.event_rules:
                if event_en.lower() in k.lower() or k.lower() in event_en.lower():
                    rules = self.event_rules[k]
                    break
        rules_text = "\n".join([f"   • {r[:100]}" for r in rules[:3]]) or "   (Нет правил)"

        # ═══ Чаты трейдеров (для объяснения) ═══
        chat_texts = []
        for idx in valid_sent_idx[:3]:
            if idx < len(self.trader_chats):
                c = self.trader_chats[idx]
                if isinstance(c, str) and len(c.strip()) > 5:
                    chat_texts.append(c.strip()[:150])
        chats_text = "\n".join([f"   💬 «{c}»" for c in chat_texts]) or "   💬 (Нет чатов)"

        # ═══ Последние 5 ═══
        last5_text = ""
        for row in hist["last5"]:
            a, f, p = row
            last5_text += f"     Ф: {a} | Пр: {f} | Пд: {p}\n"

        # ═══ Статистика истории ═══
        pct_hist = hist["pct_above"]

        # ═══ Вердикт ═══
        if fus_conf >= 65:
            verdict_emoji = "🟢"
            verdict_word = "STRONG"
        elif fus_conf >= 55:
            verdict_emoji = "🟡"
            verdict_word = "MEDIUM"
        else:
            verdict_emoji = "⚪"
            verdict_word = "WEAK"

        diff = forecast - previous
        diff_text = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"

        # Какие рыночные данные взяты из ближайшей строки
        row = self.history_df.iloc[best_idx]
        rsi_val = row.get("SP500_RSI", 0)
        vix_val = row.get("VIX_Close", 0)
        dxy_val = row.get("DXY_Close", 0)

        report = (
            f"<b>🤖 SNIPER SIGNAL: {hist['event_name']}</b>\n"
            f"   {event_ru}\n"
            f"   Валюта: {currency} | Важность: {'⭐' * importance}\n"
            f"   Прогноз: {forecast}% | Пред: {previous}% ({diff_text})\n\n"

            f"━━ 🧠 <b>PyTorch Fusion Network</b> ━━\n"
            f"   Направление: <b>{fus_dir}</b>\n"
            f"   Уверенность: <b>{fus_conf:.1f}%</b>\n\n"

            f"   📊 <b>Ветка 1 — Tabular (139 фичей):</b>\n"
            f"   RSI: {rsi_val:.0f} | VIX: {vix_val:.1f} | DXY: {dxy_val:.1f}\n"
            f"   (взято из ближайшей строки в истории)\n\n"

            f"   📚 <b>Ветка 2 — Rules / Книги TA/FA:</b>\n"
            f"{rules_text}\n\n"

            f"   🗣️ <b>Ветка 3 — Сентимент (чаты):</b>\n"
            f"{chats_text}\n\n"

            f"━━ 📊 <b>Историческая статистика</b> ━━\n"
            f"   Всего случаев: {hist['total']}\n"
            f"   Факт > Прогноз: {hist['sim_above']}/{hist['sim_total']} ({pct_hist:.0f}%)\n"
            f"   Средний факт: {hist['avg_actual']:.2f}%\n\n"

            f"   📋 <b>Последние 5:</b>\n"
            f"{last5_text}\n"

            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>🚦 {verdict_emoji} {verdict_word}: Факт скорее {fus_dir.split()[0]}"
            f" прогноза ({fus_conf:.0f}%)</b>\n"
            f"<b>   Мой прогноз: Факт ≈ {hist['avg_actual']:.2f}%</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return report


# ═══════════════════════════════════════════════════════
#  ПАРСЕР СООБЩЕНИЯ
# ═══════════════════════════════════════════════════════
def parse_number(text: str) -> float:
    s = text.strip().replace("%", "").replace(",", ".").replace(" ", "")
    return float(s)


def parse_message(text: str) -> dict:
    result = {}

    m = re.search(r'(?:валюта|currency)\s*[-–:=]\s*(\w+)', text, re.I)
    if m:
        result["currency"] = m.group(1).upper()

    m = re.search(r'(?:событие|event)\s*[-–:=]\s*(.+)', text, re.I)
    if m:
        result["event"] = m.group(1).strip()

    m = re.search(r'(?:важн|importance)\s*[.\-–:=]\s*(\d)', text, re.I)
    if m:
        result["importance"] = int(m.group(1))

    m = re.search(r'(?:прогноз|forecast)\s*[-–:=]\s*(-?\d[\d,. ]*)', text, re.I)
    if m:
        try:
            result["forecast"] = parse_number(m.group(1))
        except ValueError:
            pass

    # «Пред.-0,6%» — тире = разделитель, число = 0,6
    m = re.search(r'(?:пред|previous)[.\s]*[-–:=]+\s*(-?\d[\d,. ]*)', text, re.I)
    if m:
        try:
            result["previous"] = parse_number(m.group(1))
        except ValueError:
            pass

    return result


# ═══════════════════════════════════════════════════════
#  TELEGRAM BOT
# ═══════════════════════════════════════════════════════
router = Router()
brain  = MalishBrain()


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я <b>Малыш</b> — бот-снайпер.\n\n"
        "🧠 Мой мозг — <b>PyTorch Fusion Network</b>:\n"
        "  📊 139 фичей (RSI, VIX, MACD, Volume...)\n"
        "  📚 Книги по TA/FA (embeddings)\n"
        "  🗣️ 30,000 диалогов трейдеров (embeddings)\n\n"
        "Пришли новость:\n"
        "<pre>"
        "Валюта-USD\n"
        "Событие-CPI (м/м)\n"
        "Важн.-3 звезды\n"
        "Прогноз-0,4%\n"
        "Пред.-0,6%"
        "</pre>",
        parse_mode="HTML",
    )


@router.message(F.text)
async def handle_message(message: Message):
    text = message.text or ""
    data = parse_message(text)

    if "event" not in data or "forecast" not in data:
        await message.answer(
            "❌ Нужно минимум: <b>Событие</b>, <b>Прогноз</b>, <b>Пред.</b>\n"
            "Нажми /start для примера.",
            parse_mode="HTML",
        )
        return

    currency   = data.get("currency", "USD")
    importance = data.get("importance", 3)
    forecast   = data["forecast"]
    previous   = data.get("previous", forecast)
    event_ru   = data["event"]

    await message.answer("⏳ Запускаю нейросеть (3 ветки)...")

    try:
        report = brain.predict(event_ru, currency, importance, forecast, previous)
        await message.answer(report, parse_mode="HTML")
    except Exception as e:
        logger.exception("Prediction error")
        await message.answer(f"❌ Ошибка:\n<code>{e}</code>", parse_mode="HTML")


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
async def main():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN не задан!  export BOT_TOKEN='...'")
        return

    brain.load()

    bot = Bot(token=BOT_TOKEN)
    dp  = Dispatcher()
    dp.include_router(router)

    print("🚀 Малыш v4.0 (Fusion) запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
