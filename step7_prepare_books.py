#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 7: БАЗА ЗНАНИЙ ПО ФУНДАМЕНТАЛЬНОМУ И ТЕХ.АНАЛИЗУ ║
║  Правила из открытых источников (Investopedia, BabyPips)║
║  → data/knowledge_base.json                             ║
╚══════════════════════════════════════════════════════════╝
"""
import json
import os

OUTPUT_FILE = "data/knowledge_base.json"


def create_knowledge_base():
    """
    Создаём базу знаний из открытых источников по FA и TA.
    Каждое правило = текст + теги (для какого типа событий/индикаторов).
    """
    
    rules = []
    
    # ═══════════════════════════════════════════════════
    # 📚 ФУНДАМЕНТАЛЬНЫЙ АНАЛИЗ (FA)
    # Источники: Investopedia, BabyPips, Federal Reserve Education
    # ═══════════════════════════════════════════════════
    
    # --- CPI (Consumer Price Index) ---
    rules.extend([
        {
            "text": "CPI measures the average change in prices paid by consumers for a basket of goods and services. Higher than expected CPI typically strengthens the currency as it signals tighter monetary policy ahead.",
            "text_ru": "CPI измеряет среднее изменение цен потребительской корзины. CPI выше прогноза обычно укрепляет валюту, так как сигнализирует об ужесточении монетарной политики.",
            "events": ["CPI", "Core CPI", "CPI Flash Estimate"],
            "category": "fundamental",
            "impact": "high"
        },
        {
            "text": "When CPI consistently beats expectations for 3+ months, the central bank is likely to raise interest rates. This creates a strong bullish trend for the currency.",
            "text_ru": "Когда CPI стабильно превышает прогнозы 3+ месяца подряд, центральный банк скорее всего повысит ставки. Это создаёт сильный бычий тренд для валюты.",
            "events": ["CPI", "Core CPI"],
            "category": "fundamental",
            "impact": "high"
        },
        {
            "text": "If CPI is below expectations but the labor market remains strong, the currency reaction may be muted as the central bank weighs both factors.",
            "text_ru": "Если CPI ниже прогноза, но рынок труда остаётся сильным, реакция валюты может быть сдержанной, так как ЦБ взвешивает оба фактора.",
            "events": ["CPI", "Non-Farm Employment Change"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- NFP (Non-Farm Payrolls) ---
    rules.extend([
        {
            "text": "Non-Farm Payrolls is the most important employment indicator. Better than expected NFP strongly supports the US dollar. The market typically reacts within seconds of release.",
            "text_ru": "Non-Farm Payrolls — важнейший индикатор занятости. NFP лучше прогноза сильно поддерживает доллар. Рынок реагирует в течение секунд после выхода.",
            "events": ["Non-Farm Employment Change", "ADP Non-Farm Employment Change"],
            "category": "fundamental",
            "impact": "high"
        },
        {
            "text": "NFP revisions matter as much as the headline number. If the previous month is revised significantly downward, even a strong current number may not fully support the currency.",
            "text_ru": "Пересмотры NFP важны не менее самого числа. Если предыдущий месяц значительно пересмотрен вниз, даже сильное текущее число может не полностью поддержать валюту.",
            "events": ["Non-Farm Employment Change"],
            "category": "fundamental",
            "impact": "high"
        },
        {
            "text": "ADP employment report is released two days before NFP and serves as a leading indicator. However, ADP and NFP often diverge significantly.",
            "text_ru": "Отчёт ADP выходит за два дня до NFP и служит опережающим индикатором. Однако ADP и NFP часто значительно расходятся.",
            "events": ["ADP Non-Farm Employment Change", "Non-Farm Employment Change"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- PMI (Purchasing Managers Index) ---
    rules.extend([
        {
            "text": "PMI above 50 indicates economic expansion, below 50 indicates contraction. The distance from 50 matters more than the absolute value. PMI above 55 signals strong growth.",
            "text_ru": "PMI выше 50 указывает на расширение экономики, ниже 50 — на сокращение. Расстояние от 50 важнее абсолютного значения. PMI выше 55 сигнализирует о сильном росте.",
            "events": ["Manufacturing PMI", "Services PMI", "Flash Manufacturing PMI", "Final Manufacturing PMI", "ISM Manufacturing PMI"],
            "category": "fundamental",
            "impact": "medium"
        },
        {
            "text": "Flash PMI releases have more market impact than final PMI because they provide the first look at economic activity for the month.",
            "text_ru": "Предварительный PMI оказывает большее влияние на рынок чем финальный, т.к. дает первый взгляд на экономическую активность за месяц.",
            "events": ["Flash Manufacturing PMI", "Flash Services PMI"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- Interest Rates & Central Banks ---
    rules.extend([
        {
            "text": "Interest rate decisions are the most impactful economic events. A surprise rate hike strongly strengthens the currency. Forward guidance matters as much as the actual decision.",
            "text_ru": "Решения по процентным ставкам — самые влиятельные экономические события. Неожиданное повышение ставки сильно укрепляет валюту. Прогнозы ЦБ важны не менее самого решения.",
            "events": ["Federal Funds Rate", "Minimum Bid Rate", "Official Bank Rate"],
            "category": "fundamental",
            "impact": "high"
        },
        {
            "text": "When unemployment is low and CPI is high, the probability of a rate hike increases. Watch for hawkish language in FOMC statements.",
            "text_ru": "Когда безработица низкая, а CPI высокий, вероятность повышения ставки растёт. Следите за ястребиной риторикой в заявлениях FOMC.",
            "events": ["Unemployment Rate", "CPI", "Federal Funds Rate", "FOMC Statement"],
            "category": "fundamental",
            "impact": "high"
        },
    ])
    
    # --- Trade Balance ---
    rules.extend([
        {
            "text": "A larger trade deficit than expected is generally bearish for the currency, as it means more money is flowing out of the country to buy foreign goods.",
            "text_ru": "Торговый дефицит больше ожидаемого обычно медвежий для валюты: больше денег утекает из страны на покупку иностранных товаров.",
            "events": ["Trade Balance"],
            "category": "fundamental",
            "impact": "medium"
        },
        {
            "text": "Oil price changes directly impact trade balance. For oil-importing nations, higher oil prices widen the trade deficit.",
            "text_ru": "Изменения цен на нефть напрямую влияют на торговый баланс. Для стран-импортёров нефти рост цен увеличивает торговый дефицит.",
            "events": ["Trade Balance", "Crude Oil Inventories"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- Retail Sales ---
    rules.extend([
        {
            "text": "Retail sales reflect consumer spending, which accounts for about 70% of GDP. Strong retail sales support the currency by indicating robust economic growth.",
            "text_ru": "Розничные продажи отражают потребительские расходы — около 70% ВВП. Сильные продажи поддерживают валюту, указывая на устойчивый экономический рост.",
            "events": ["Retail Sales", "Core Retail Sales"],
            "category": "fundamental",
            "impact": "medium"
        },
        {
            "text": "Holiday season retail sales (November-December) tend to be higher. January often shows a decline, which is seasonal and should not be overinterpreted.",
            "text_ru": "Праздничные розничные продажи (ноябрь-декабрь) обычно выше. Январь часто показывает снижение — это сезонность, не стоит переоценивать.",
            "events": ["Retail Sales"],
            "category": "fundamental",
            "impact": "low"
        },
    ])
    
    # --- GDP ---
    rules.extend([
        {
            "text": "GDP is the broadest measure of economic activity. However, by the time GDP is released, the market has already priced in most of the information from earlier indicators.",
            "text_ru": "ВВП — самый широкий показатель экономической активности. Но к моменту выхода ВВП рынок уже учёл большую часть информации из ранних индикаторов.",
            "events": ["GDP", "Advance GDP", "Prelim GDP", "Final GDP"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- Unemployment ---
    rules.extend([
        {
            "text": "Unemployment Claims is a weekly indicator. A consistently declining trend signals improving labor market. Sudden spikes above 300K signal economic distress.",
            "text_ru": "Заявки на пособие по безработице — еженедельный индикатор. Устойчивый снижающийся тренд сигнализирует об улучшении рынка труда. Резкие скачки выше 300K — сигнал экономических проблем.",
            "events": ["Unemployment Claims"],
            "category": "fundamental",
            "impact": "medium"
        },
        {
            "text": "Unemployment rate below 4% is considered full employment for the US. At this level, wage growth accelerates and inflation pressures build.",
            "text_ru": "Безработица ниже 4% считается полной занятостью для США. На этом уровне рост зарплат ускоряется и инфляционное давление нарастает.",
            "events": ["Unemployment Rate", "Average Hourly Earnings"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- Oil & Energy ---
    rules.extend([
        {
            "text": "Crude Oil Inventories: a build larger than expected is bearish for oil prices, while a draw larger than expected is bullish. Oil prices inversely correlate with the US dollar.",
            "text_ru": "Запасы нефти: запасы больше ожидаемых медвежьи для нефти, запасы меньше ожидаемых — бычьи. Цены на нефть обратно коррелируют с долларом.",
            "events": ["Crude Oil Inventories", "Natural Gas Storage"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- Consumer Confidence ---
    rules.extend([
        {
            "text": "Consumer confidence indexes are forward-looking indicators. High confidence leads to more spending, which supports GDP growth and the currency.",
            "text_ru": "Индексы потребительского доверия — опережающие индикаторы. Высокое доверие ведёт к росту расходов, что поддерживает ВВП и валюту.",
            "events": ["Consumer Confidence", "UoM Consumer Sentiment", "CB Consumer Confidence"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # --- PPI ---
    rules.extend([
        {
            "text": "PPI measures wholesale inflation. Rising PPI often leads to rising CPI 1-2 months later, as producers pass costs to consumers.",
            "text_ru": "PPI измеряет оптовую инфляцию. Рост PPI часто приводит к росту CPI через 1-2 месяца, т.к. производители перекладывают издержки на потребителей.",
            "events": ["PPI", "Core PPI"],
            "category": "fundamental",
            "impact": "medium"
        },
    ])
    
    # ═══════════════════════════════════════════════════
    # 📈 ТЕХНИЧЕСКИЙ АНАЛИЗ (TA)
    # Источники: TradingView, Investopedia, Technical Analysis of Financial Markets
    # ═══════════════════════════════════════════════════
    
    # --- RSI ---
    rules.extend([
        {
            "text": "RSI above 70 indicates overbought conditions. When RSI is above 70 at the time of a positive economic release, the upside reaction may be limited due to exhaustion.",
            "text_ru": "RSI выше 70 указывает на перекупленность. Когда RSI выше 70 в момент позитивных данных, рост может быть ограничен из-за истощения.",
            "events": ["*"],
            "category": "technical",
            "indicator": "RSI",
            "impact": "medium"
        },
        {
            "text": "RSI below 30 indicates oversold conditions. Positive economic data during oversold RSI often triggers a strong rebound as shorts cover their positions.",
            "text_ru": "RSI ниже 30 указывает на перепроданность. Позитивные данные при перепроданном RSI часто вызывают сильный отскок, т.к. шорты закрывают позиции.",
            "events": ["*"],
            "category": "technical",
            "indicator": "RSI",
            "impact": "medium"
        },
        {
            "text": "RSI divergence is a powerful signal. If prices make new highs but RSI makes lower highs, a reversal is likely. Economic data releases often act as catalysts for such reversals.",
            "text_ru": "Дивергенция RSI — мощный сигнал. Если цены обновляют максимумы, а RSI делает более низкие максимумы, разворот вероятен. Экономические данные часто становятся катализатором таких разворотов.",
            "events": ["*"],
            "category": "technical",
            "indicator": "RSI",
            "impact": "high"
        },
    ])
    
    # --- MACD ---
    rules.extend([
        {
            "text": "MACD histogram crossing zero line is a trend change signal. Positive histogram during news release enhances bullish reaction.",
            "text_ru": "Пересечение гистограммы MACD нулевой линии — сигнал смены тренда. Позитивная гистограмма в момент новости усиливает бычью реакцию.",
            "events": ["*"],
            "category": "technical",
            "indicator": "MACD",
            "impact": "medium"
        },
        {
            "text": "MACD signal line crossover combined with positive economic data creates a strong momentum signal. Both technical and fundamental factors align.",
            "text_ru": "Пересечение сигнальной линии MACD в сочетании с позитивными данными создаёт сильный импульсный сигнал. Технические и фундаментальные факторы совпадают.",
            "events": ["*"],
            "category": "technical",
            "indicator": "MACD",
            "impact": "high"
        },
    ])
    
    # --- Bollinger Bands ---
    rules.extend([
        {
            "text": "Price at the upper Bollinger Band with expanding bandwidth suggests strong uptrend. News-driven breakouts above the upper band often lead to continuation.",
            "text_ru": "Цена у верхней полосы Боллинджера с расширяющейся шириной указывает на сильный восходящий тренд. Прорывы на новостях выше верхней полосы часто продолжаются.",
            "events": ["*"],
            "category": "technical",
            "indicator": "Bollinger",
            "impact": "medium"
        },
        {
            "text": "Bollinger Band squeeze (narrow width) before a major economic release indicates a potential explosive move. The direction will be determined by the data.",
            "text_ru": "Сжатие полос Боллинджера (узкая ширина) перед важными данными указывает на потенциальный взрывной ход. Направление определят данные.",
            "events": ["*"],
            "category": "technical",
            "indicator": "Bollinger",
            "impact": "high"
        },
    ])
    
    # --- SMA ---
    rules.extend([
        {
            "text": "Price above both SMA20 and SMA50 with SMA20 above SMA50 (Golden Cross) is a bullish signal. Positive economic data in this context tends to produce larger moves.",
            "text_ru": "Цена выше SMA20 и SMA50, при SMA20 выше SMA50 (Золотой крест) — бычий сигнал. Позитивные данные в этом контексте обычно создают более сильные движения.",
            "events": ["*"],
            "category": "technical",
            "indicator": "SMA",
            "impact": "medium"
        },
        {
            "text": "SMA50 acts as dynamic support in uptrends and resistance in downtrends. Economic releases near SMA50 level often determine the next directional move.",
            "text_ru": "SMA50 действует как динамическая поддержка в восходящих трендах и сопротивление в нисходящих. Экономические данные вблизи SMA50 часто определяют следующее направление.",
            "events": ["*"],
            "category": "technical",
            "indicator": "SMA",
            "impact": "medium"
        },
    ])
    
    # --- VIX ---
    rules.extend([
        {
            "text": "VIX above 25 indicates high market fear. Economic data worse than expected during high VIX tends to cause amplified negative reactions, while positive surprises cause relief rallies.",
            "text_ru": "VIX выше 25 указывает на высокий страх рынка. Данные хуже ожиданий при высоком VIX вызывают усиленные негативные реакции, а позитивные сюрпризы — ралли облегчения.",
            "events": ["*"],
            "category": "technical",
            "indicator": "VIX",
            "impact": "high"
        },
        {
            "text": "Low VIX below 15 often precedes complacency. Negative economic surprises during low VIX can trigger outsized selling as markets were not prepared for bad news.",
            "text_ru": "Низкий VIX ниже 15 часто предшествует самоуспокоенности. Негативные экономические сюрпризы при низком VIX могут вызвать чрезмерные продажи, т.к. рынки были не готовы к плохим новостям.",
            "events": ["*"],
            "category": "technical",
            "indicator": "VIX",
            "impact": "high"
        },
    ])
    
    # --- Volume Analysis ---
    rules.extend([
        {
            "text": "Volume spikes before economic releases indicate institutional positioning. Smart money often positions before the data comes out based on their own estimates.",
            "text_ru": "Всплески объёма перед экономическими данными указывают на позиционирование институционалов. Умные деньги часто открывают позиции до выхода данных на основе собственных оценок.",
            "events": ["*"],
            "category": "volume",
            "indicator": "Volume",
            "impact": "high"
        },
        {
            "text": "On-Balance Volume divergence with price suggests the underlying trend is weakening. If OBV falls while price rises, a negative economic surprise may trigger a sharp reversal.",
            "text_ru": "Дивергенция OBV с ценой указывает на ослабление базового тренда. Если OBV падает а цена растёт, негативный экономический сюрприз может вызвать резкий разворот.",
            "events": ["*"],
            "category": "volume",
            "indicator": "OBV",
            "impact": "medium"
        },
        {
            "text": "Money Flow Index above 80 combined with worse-than-expected data creates a strong sell signal. MFI below 20 with better-than-expected data creates a strong buy signal.",
            "text_ru": "Money Flow Index выше 80 в сочетании с данными хуже ожиданий создаёт сильный сигнал продажи. MFI ниже 20 с данными лучше ожиданий — сильный сигнал покупки.",
            "events": ["*"],
            "category": "volume",
            "indicator": "MFI",
            "impact": "medium"
        },
    ])
    
    # --- Wave / Trend Analysis ---
    rules.extend([
        {
            "text": "In a strong uptrend with 5+ consecutive positive days, positive economic data extends the trend. However, extremely long streaks (10+) increase reversal probability.",
            "text_ru": "В сильном восходящем тренде с 5+ последовательными позитивными днями, позитивные данные продлевают тренд. Однако экстремально длинные серии (10+) увеличивают вероятность разворота.",
            "events": ["*"],
            "category": "wave",
            "indicator": "Trend",
            "impact": "medium"
        },
        {
            "text": "Price near 20-day swing highs during positive economic release often leads to breakout. Price near swing lows during negative data accelerates the decline.",
            "text_ru": "Цена вблизи 20-дневных swing максимумов при позитивных данных часто ведёт к прорыву. Цена вблизи swing минимумов при негативных данных ускоряет снижение.",
            "events": ["*"],
            "category": "wave",
            "indicator": "Swing",
            "impact": "medium"
        },
        {
            "text": "High wave count (many reversals) in the past 20 days indicates choppy/range-bound market. In such conditions, economic data reactions are often short-lived.",
            "text_ru": "Высокое количество волн (много разворотов) за 20 дней указывает на диапазонный рынок. В таких условиях реакции на экономические данные часто краткосрочны.",
            "events": ["*"],
            "category": "wave",
            "indicator": "Wave",
            "impact": "low"
        },
    ])
    
    # --- Cross-market signals ---
    rules.extend([
        {
            "text": "When S&P500 and DXY move in opposite directions (risk-on: S&P up, DXY down), positive US economic data may strengthen both as the economy looks strong enough for growth AND a strong dollar.",
            "text_ru": "Когда S&P500 и DXY движутся в противоположных направлениях (risk-on: S&P растёт, DXY падает), позитивные данные по США могут усилить оба: экономика достаточно сильна для роста И сильного доллара.",
            "events": ["*"],
            "category": "cross_market",
            "impact": "high"
        },
        {
            "text": "Rising US10Y yields with falling VIX create a goldilocks environment where positive economic data has maximum positive impact on the dollar.",
            "text_ru": "Растущая доходность US10Y при падающем VIX создают идеальную среду, где позитивные данные оказывают максимальное позитивное влияние на доллар.",
            "events": ["*"],
            "category": "cross_market",
            "impact": "high"
        },
        {
            "text": "When EUR economic data is strong the same week as USD data, the EUR/USD reaction depends on the relative surprise. Always compare the magnitude of surprises.",
            "text_ru": "Когда данные по EUR сильные на той же неделе что и данные по USD, реакция EUR/USD зависит от относительного сюрприза. Всегда сравнивайте величину сюрпризов.",
            "events": ["*"],
            "category": "cross_market",
            "impact": "medium"
        },
    ])
    
    # --- Seasonality & Timing ---
    rules.extend([
        {
            "text": "First Friday of the month (NFP day) is the most volatile day. Position sizing should be reduced. The initial reaction often reverses within 30 minutes.",
            "text_ru": "Первая пятница месяца (день NFP) — самый волатильный день. Размер позиций стоит уменьшить. Первоначальная реакция часто разворачивается в течение 30 минут.",
            "events": ["Non-Farm Employment Change"],
            "category": "timing",
            "impact": "high"
        },
        {
            "text": "Economic data released during Asian session (early US morning) tends to have muted reactions due to lower liquidity. Peak reactions happen during London-New York overlap.",
            "text_ru": "Экономические данные во время азиатской сессии обычно вызывают сдержанные реакции из-за низкой ликвидности. Пиковые реакции — во время перекрытия Лондон-Нью-Йорк.",
            "events": ["*"],
            "category": "timing",
            "impact": "low"
        },
        {
            "text": "End-of-quarter data releases (March, June, September, December) tend to show window-dressing effects. GDP and PMI numbers can appear stronger due to rebalancing flows.",
            "text_ru": "Данные на конце квартала (март, июнь, сентябрь, декабрь) подвержены эффекту «приукрашивания». ВВП и PMI могут выглядеть сильнее из-за перебалансировки портфелей.",
            "events": ["GDP", "Manufacturing PMI"],
            "category": "timing",
            "impact": "low"
        },
    ])
    
    # Сохраняем
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    
    # Статистика
    categories = {}
    for r in rules:
        cat = r.get('category', 'other')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ БАЗА ЗНАНИЙ СОЗДАНА                                 ║
╚══════════════════════════════════════════════════════════╝

  📁 Файл: {OUTPUT_FILE}
  📚 Всего правил: {len(rules)}
  
  📊 По категориям:
""")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        icon = {"fundamental": "📰", "technical": "📈", "volume": "🔊", 
                "wave": "🌊", "cross_market": "🔗", "timing": "⏰"}.get(cat, "📋")
        print(f"     {icon} {cat}: {cnt} правил")
    
    print(f"\n  Следующий шаг: python3 step8_create_embeddings.py")


if __name__ == "__main__":
    create_knowledge_base()
