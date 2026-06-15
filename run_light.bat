@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ===================================================
echo GPB MER - Запуск приложения (Легкий режим Mock)
echo ===================================================
if not exist .venv_light\Scripts\activate.bat (
    echo [ERROR] Виртуальное окружение .venv_light не найдено. Сначала запустите install_light.bat.
    pause
    exit /b 1
)
call .venv_light\Scripts\activate.bat
set MOCK_MODE=True
python app.py
pause
