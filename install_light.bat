@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ===================================================
echo GPB MER - Создание виртуального окружения (Легкое)
echo ===================================================
python -m venv .venv_light
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось создать виртуальное окружение. Убедитесь, что Python установлен.
    pause
    exit /b %errorlevel%
)
echo [OK] Виртуальное окружение .venv_light создано.
echo Активация окружения и установка зависимостей...
call .venv_light\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements_light.txt
if %errorlevel% neq 0 (
    echo [ERROR] Ошибка установки зависимостей.
    pause
    exit /b %errorlevel%
)
echo [SUCCESS] Установка завершена успешно! Используйте run_light.bat для запуска.
pause
