@echo off
chcp 65001 > nul
echo ===================================================
echo GPB MER - Запуск приложения (Полный режим с GPU)
echo ===================================================
if not exist .venv_full\Scripts\activate.bat (
    echo [ERROR] Виртуальное окружение .venv_full не найдено. Сначала запустите install_full.bat.
    pause
    exit /b 1
)
call .venv_full\Scripts\activate.bat
set MOCK_MODE=False
python app.py
pause
