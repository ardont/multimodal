@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ===================================================
echo GPB MER - Запуск сервера от имени Администратора
echo ===================================================

:: Проверка прав администратора
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Запрос прав администратора...
    powershell -Command "Start-Process '%~dpnx0' -Verb RunAs"
    exit /b
)

echo [OK] Запущено с правами Администратора.

echo [1/3] Проверка и настройка Брандмауэра Windows...
powershell -Command "if (-not (Get-NetFirewallRule -DisplayName 'GPB MER Port 7860' -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName 'GPB MER Port 7860' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7860 | Out-Null; Write-Host '[OK] Правило брандмауэра для порта 7860 успешно создано.' } else { Write-Host '[OK] Правило брандмауэра уже существует.' }"

echo [2/3] Определение сетевых адресов этого компьютера...
echo ---------------------------------------------------
echo Доступные IP-адреса для подключения с ноутбука:
powershell -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | ForEach-Object { echo ('  - ' + $_.IPAddress + ' (Интерфейс: ' + $_.InterfaceAlias + ')') }"
echo ---------------------------------------------------

echo [3/3] Запуск сервера приложений...
if not exist .venv_full\Scripts\activate.bat (
    echo [ERROR] Виртуальное окружение .venv_full не найдено. Сначала запустите install_full.bat.
    pause
    exit /b 1
)

call .venv_full\Scripts\activate.bat
set MOCK_MODE=False
python app.py
pause
