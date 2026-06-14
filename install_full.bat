@echo off
chcp 65001 > nul
echo ===================================================
echo GPB MER - Создание виртуального окружения (Полное)
echo ===================================================
python -m venv .venv_full
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось создать виртуальное окружение. Убедитесь, что Python установлен.
    pause
    exit /b %errorlevel%
)
echo [OK] Виртуальное окружение .venv_full создано.
echo Активация окружения и установка зависимостей...
call .venv_full\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Ошибка установки зависимостей.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Установка завершена успешно! Используйте run_full.bat для запуска.
pause
