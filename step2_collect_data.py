#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 2: СБОР ДАННЫХ (Скачиваем историю новостей)       ║
╚══════════════════════════════════════════════════════════╝

Запуск: python3 step2_collect_data.py

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ:
  Скачивает экономический календарь с сайта investing.com.
  Это таблица с 15 000+ новостей за 2000-2026 годы.

  Каждая строка — одна новость:
    Дата | Время | Валюта | Событие | Важность | Факт | Прогноз | Пред.

ЗАЧЕМ:
  Без данных модель не может учиться.
  Данные = учебник для нашего "стажёра".

РЕЗУЛЬТАТ:
  Файл data/raw_calendar.csv — сырые данные.
"""
import asyncio
import csv
import os
import random
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════
# 📚 СЛОВАРИК для новичка:
#
# CSV — таблица в текстовом формате (Excel без Excel)
# async — код, который "ждёт" ответа от сайта
# Playwright — робот-браузер, который открывает сайты
# Scraping — автоматическое извлечение данных с сайтов
# ═══════════════════════════════════════════════════════

# Куда сохраняем данные
DATA_DIR = "data"
OUTPUT_FILE = f"{DATA_DIR}/raw_calendar.csv"

# Диапазон дат
START_DATE = datetime(2000, 1, 1)
END_DATE = datetime(2026, 2, 1)

async def collect_data():
    """
    Основная функция сбора данных.
    
    Как работает:
    1. Открывает браузер (Playwright)
    2. Заходит на investing.com/economic-calendar
    3. Листает по неделям с 2000 по 2026 год
    4. Извлекает таблицу новостей
    5. Сохраняет в CSV файл
    """
    from playwright.async_api import async_playwright
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("╔══════════════════════════════════════════════════╗")
    print("║  📥 СБОР ДАННЫХ: Экономический календарь        ║")
    print("║  Источник: investing.com                        ║")
    print("║  Период: 2000 — 2026                            ║")
    print("╚══════════════════════════════════════════════════╝")
    
    # Открываем CSV файл для записи
    csvfile = open(OUTPUT_FILE, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Time', 'Currency', 'Event', 'Importance', 'Actual', 'Forecast', 'Previous'])
    
    total_rows = 0
    
    async with async_playwright() as p:
        # ═══════════════════════════════
        # Запускаем ВИДИМЫЙ браузер
        # (headless=False — чтобы ты видел, что происходит)
        # ═══════════════════════════════
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        current_date = START_DATE
        
        while current_date < END_DATE:
            # Формируем URL с датой
            date_str = current_date.strftime('%Y/%m/%d')
            url = f"https://www.investing.com/economic-calendar/?startDate={date_str}"
            
            try:
                await page.goto(url, timeout=30000)
                await page.wait_for_timeout(random.randint(2000, 4000))
                
                # ═══════════════════════════════
                # Извлекаем данные из таблицы
                # JavaScript код выполняется В БРАУЗЕРЕ
                # и возвращает нам массив строк
                # ═══════════════════════════════
                rows = await page.evaluate("""
                    (async () => {
                        const rows = [];
                        const table = document.querySelectorAll('tr.js-event-item, tr[data-event-datetime]');
                        table.forEach(tr => {
                            try {
                                const dateEl = tr.getAttribute('data-event-datetime') || '';
                                const time = tr.querySelector('td.time')?.textContent?.trim() || '';
                                const currency = tr.querySelector('td.flagCur span')?.textContent?.trim() || 
                                                 tr.querySelector('td.flagCur')?.textContent?.trim() || '';
                                const event = tr.querySelector('td.event a')?.textContent?.trim() || 
                                              tr.querySelector('td.event')?.textContent?.trim() || '';
                                const bulls = tr.querySelectorAll('td.sentiment i.grayFullBullishIcon').length;
                                const actual = tr.querySelector('td.act, td.bold')?.textContent?.trim() || '';
                                const forecast = tr.querySelector('td.fore')?.textContent?.trim() || '';
                                const previous = tr.querySelector('td.prev')?.textContent?.trim() || '';
                                
                                if (event) {
                                    rows.push([dateEl, time, currency, event, bulls, actual, forecast, previous]);
                                }
                            } catch(e) {}
                        });
                        return rows;
                    })()
                """)
                
                if rows:
                    for row in rows:
                        # Добавляем дату если её нет в данных
                        if not row[0]:
                            row[0] = current_date.strftime('%Y-%m-%d')
                        writer.writerow(row)
                        total_rows += 1
                
                print(f"   📅 {date_str}: +{len(rows) if rows else 0} новостей (всего: {total_rows})")
                
            except Exception as e:
                print(f"   ⚠️ {date_str}: ошибка ({str(e)[:50]})")
            
            # Переходим к следующей неделе
            current_date += timedelta(days=7)
            
            # Пауза, чтобы сайт не заблокировал
            await page.wait_for_timeout(random.randint(1000, 3000))
    
    csvfile.close()
    
    print(f"""
╔══════════════════════════════════════════════════╗
║  ✅ СБОР ДАННЫХ ЗАВЕРШЁН!                       ║
║  Файл: {OUTPUT_FILE:<40} ║
║  Строк: {total_rows:<39} ║
╚══════════════════════════════════════════════════╝

  Следующий шаг: python3 step3_clean_data.py
""")

if __name__ == "__main__":
    asyncio.run(collect_data())
