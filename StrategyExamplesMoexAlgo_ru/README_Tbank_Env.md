# Инструкция по настройке переменных окружения для Tbank (Тинькофф Инвестиции)

## Переменные окружения

### INVEST_TOKEN
Токен для доступа к API Тинькофф Инвестиций.

**Для sandbox режима (тестовый токен):**
```bash
export INVEST_TOKEN="t.ZLtpCN0pOiGj8WbOU0xxGgpCWxBH5vmnYH-hzvgXQesS04yGMtEiw1tzJevGZox1r6nVMXi0z0QMO3BaRH7lBA"
```

### INVEST_ACCOUNT_ID (опционально)
ID счета для торговли. Если не указан, будет использовано значение из файла конфигурации `my_config/Config_Tbank.py`.

```bash
export INVEST_ACCOUNT_ID="20000000000"  # замените на ваш ID счета
```

## Установка переменных окружения

### Linux / macOS

Временная установка (действует до закрытия терминала):
```bash
export INVEST_TOKEN="t.ZLtpCN0pOiGj8WbOU0xxGgpCWxBH5vmnYH-hzvgXQesS04yGMtEiw1tzJevGZox1r6nVMXi0z0QMO3BaRH7lBA"
export INVEST_ACCOUNT_ID="20000000000"
```

Постоянная установка (добавить в ~/.bashrc или ~/.zshrc):
```bash
echo 'export INVEST_TOKEN="t.ZLtpCN0pOiGj8WbOU0xxGgpCWxBH5vmnYH-hzvgXQesS04yGMtEiw1tzJevGZox1r6nVMXi0z0QMO3BaRH7lBA"' >> ~/.bashrc
echo 'export INVEST_ACCOUNT_ID="20000000000"' >> ~/.bashrc
source ~/.bashrc
```

### Windows (PowerShell)

Временная установка:
```powershell
$env:INVEST_TOKEN="t.ZLtpCN0pOiGj8WbOU0xxGgpCWxBH5vmnYH-hzvgXQesS04yGMtEiw1tzJevGZox1r6nVMXi0z0QMO3BaRH7lBA"
$env:INVEST_ACCOUNT_ID="20000000000"
```

Постоянная установка (для текущего пользователя):
```powershell
[Environment]::SetEnvironmentVariable("INVEST_TOKEN", "t.ZLtpCN0pOiGj8WbOU0xxGgpCWxBH5vmnYH-hzvgXQesS04yGMtEiw1tzJevGZox1r6nVMXi0z0QMO3BaRH7lBA", "User")
[Environment]::SetEnvironmentVariable("INVEST_ACCOUNT_ID", "20000000000", "User")
```

### Windows (cmd)

Временная установка:
```cmd
set INVEST_TOKEN=t.ZLtpCN0pOiGj8WbOU0xxGgpCWxBH5vmnYH-hzvgXQesS04yGMtEiw1tzJevGZox1r6nVMXi0z0QMO3BaRH7lBA
set INVEST_ACCOUNT_ID=20000000000
```

## Запуск скрипта

После установки переменных окружения запустите скрипт:

```bash
python "01 - Live Trade - broker Tbank.py"
```

## Проверка установки переменных

Проверить установленные переменные можно командой:

```bash
echo $INVEST_TOKEN  # Linux/macOS
echo %INVEST_TOKEN%  # Windows cmd
echo $env:INVEST_TOKEN  # Windows PowerShell
```

## Безопасность

⚠️ **Внимание!** Никогда не коммитьте токены и чувствительные данные в систему контроля версий (git).
Используйте переменные окружения или файлы `.env` (которые должны быть добавлены в `.gitignore`).

Для production среды используйте secure vault или secrets manager.
