@echo off
chcp 65001 > nul
echo ===================================================
echo GPB MER - Автоматическое Обновление Проекта
echo ===================================================

echo [1/5] Сброс блокировок Git...
if exist .git\index.lock (
    echo Найдена блокировка .git/index.lock. Удаление...
    del /f /q .git\index.lock
    if %errorlevel% neq 0 (
        echo [ERROR] Не удалось удалить .git/index.lock. Убедитесь, что процессы git завершены.
    ) else (
        echo [OK] Блокировка .git/index.lock удалена.
    )
) else (
    echo [OK] Блокировок Git не обнаружено.
)

echo [2/5] Настройка Git для работы на сетевых дисках...
git config core.fsync false
git config core.fsyncObjectFiles false
if %errorlevel% neq 0 (
    echo [WARNING] Не удалось применить настройки Git.
) else (
    echo [OK] Настройки fsync оптимизированы.
)

echo [3/5] Получение обновлений из репозитория...
git fetch origin
if %errorlevel% neq 0 (
    echo [ERROR] Ошибка при выполнении git fetch. Проверьте интернет-соединение.
    pause
    exit /b %errorlevel%
)

git reset --hard origin/main
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось сбросить состояние до origin/main.
    pause
    exit /b %errorlevel%
)
echo [OK] Локальные файлы успешно обновлены до последней версии с GitHub.

echo [4/5] Обновление виртуального окружения...
set "VENV_DIR="
if exist .venv_full (
    set "VENV_DIR=.venv_full"
    set "REQ_FILE=requirements.txt"
) else if exist .venv_light (
    set "VENV_DIR=.venv_light"
    set "REQ_FILE=requirements_light.txt"
)

if not defined VENV_DIR (
    echo [WARNING] Виртуальное окружение (.venv_full или .venv_light) не найдено.
    echo Запустите install_full.bat или install_light.bat для установки.
) else (
    echo Обнаружено окружение: %VENV_DIR%
    echo Активация окружения...
    call %VENV_DIR%\Scripts\activate.bat
    
    echo Удаление проблемного пакета hf_xet (если он установлен)...
    pip uninstall -y hf_xet > nul 2>&1
    
    echo Установка/обновление зависимостей из %REQ_FILE%...
    pip install -r %REQ_FILE%
    if %errorlevel% neq 0 (
        echo [ERROR] Ошибка при установке зависимостей.
    ) else (
        echo [OK] Зависимости успешно обновлены.
    )
)

echo [5/5] Очистка временных файлов...
git clean -df > nul 2>&1
echo [OK] Временные файлы очищены.

echo ===================================================
echo [SUCCESS] Обновление проекта завершено успешно!
echo ===================================================
pause
