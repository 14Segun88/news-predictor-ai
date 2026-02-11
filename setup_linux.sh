#!/bin/bash

echo "🚀 Начинаем установку проекта News Predictor..."

# 1. Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Ошибка: Python 3 не найден. Установите Python 3."
    exit 1
fi

# 2. Создание виртуального окружения
echo "📦 Создаем виртуальное окружение (venv)..."
python3 -m venv venv
source venv/bin/activate

# 3. Обновление pip
echo "⬇️ Обновляем pip..."
pip install --upgrade pip

# 4. Установка зависимостей
echo "📚 Устанавливаем библиотеки (это может занять время)..."
pip install -r requirements.txt

# 5. Установка браузеров Playwright
echo "🌍 Устанавливаем браузеры для Playwright..."
playwright install chromium

echo "✅ Установка завершена!"
echo ""
echo "👉 Чтобы запустить бота:"
echo "   source venv/bin/activate"
echo "   export BOT_TOKEN='ваш_токен'"
echo "   python3 step11_bot_fusion.py"
