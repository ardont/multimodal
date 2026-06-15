@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ===================================================
echo GPB MER - Жесткое принудительное обновление проекта
echo ===================================================

echo [1/5] Освобождение файлов: закрытие запущенных процессов Python...
taskkill /f /im python.exe /t >nul 2>&1
taskkill /f /im pythonw.exe /t >nul 2>&1
timeout /t 1 /nobreak >nul
echo [OK] Процессы Python остановлены.

echo [2/5] Сброс блокировок Git...
if exist .git\index.lock (
    del /f /q .git\index.lock >nul 2>&1
    echo [OK] Блокировка .git/index.lock удалена.
) else (
    echo [OK] Блокировок Git не обнаружено.
)

echo [3/5] Сброс локальных изменений и очистка...
git clean -fd >nul 2>&1
git reset --hard HEAD >nul 2>&1
echo [OK] Локальная рабочая копия очищена.

echo [4/5] Получение обновлений из репозитория GitHub...
git fetch origin
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось подключиться к GitHub для проверки обновлений.
    pause
    exit /b 1
)

git reset --hard origin/main
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось обновить рабочую копию до origin/main.
    pause
    exit /b 1
)
echo [OK] Обновление успешно стянуто.

echo [5/6] Обновление зависимостей в виртуальном окружении...
if exist .venv_full (
    echo Обнаружено окружение .venv_full. Обновление пакетов...
    call .venv_full\Scripts\activate.bat
    echo Установка PyTorch с поддержкой GPU (CUDA 12.1)...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
) else if exist .venv_light (
    echo Обнаружено окружение .venv_light. Обновление пакетов...
    call .venv_light\Scripts\activate.bat
    pip install -r requirements_light.txt
)

echo [6/6] Проверка текущей версии проекта...
echo ---------------------------------------------------
echo Текущая активная версия на этом компьютере:
git log -1 --format="Коммит: %%h%%nАвтор: %%an%%nДата:  %%ad%%nТема:  %%s"
echo ---------------------------------------------------

echo.
echo ===================================================
echo [SUCCESS] Обновление до последней версии завершено!
echo Теперь вы можете запустить run_full_admin.bat
echo ===================================================
pause
