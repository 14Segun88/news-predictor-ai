@echo off
echo 🚀 Starting News Predictor installation...

REM 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.10+
    pause
    exit /b
)

REM 2. Create venv
echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

REM 3. Install dependencies
echo 📚 Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

REM 4. Install Playwright browsers
echo 🌍 Installing Playwright browsers...
playwright install chromium

echo ✅ Installation complete!
echo.
echo 👉 To start the bot:
echo    1. venv\Scripts\activate
echo    2. set BOT_TOKEN=your_token
echo    3. python step11_bot_fusion.py
pause
