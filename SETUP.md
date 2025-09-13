# Настройка безопасности и окружения

## 1. Настройка переменных окружения

### Создание .env файла
1. Скопируйте `env_template.txt` в `.env`:
   ```bash
   cp env_template.txt .env
   ```

2. Отредактируйте `.env` файл и добавьте ваш реальный OpenAI API ключ:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

### Важно!
- **НИКОГДА** не коммитьте `.env` файл в git
- Файл `.env` уже добавлен в `.gitignore`
- Используйте `env_template.txt` как шаблон

## 2. Безопасное логирование

### Использование log_json
Вместо обычного логирования используйте `log_json` из `logging_setup.py`:

```python
from logging_setup import log_json, setup_logging

# Инициализация логгера
logger = setup_logging()

# Безопасное логирование
log_json(logger, "Processing request", 
         user_id=123, 
         request_type="search",
         # НЕ передавайте api_key, token, secret и т.д.
         )
```

### Автоматическая защита
Функция `log_json` автоматически заменяет значения для ключей, содержащих:
- `api_key`, `apikey`
- `token`
- `secret`
- `authorization`
- `password`
- `key`

На `***` для предотвращения случайного логирования секретов.

## 3. Проверка API ключа

Приложение автоматически проверяет наличие `OPENAI_API_KEY` при запуске:

```python
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")
```

## 4. Запуск приложения

1. Убедитесь, что `.env` файл создан и содержит ваш API ключ
2. Установите зависимости: `pip install -r requirements.txt`
3. Запустите приложение: `python app.py`

## 5. Примеры безопасного логирования

### ✅ Правильно:
```python
log_json(logger, "User action", 
         user_id=123, 
         action="search", 
         query_length=50)
```

### ❌ Неправильно:
```python
log_json(logger, "API call", 
         api_key="sk-...",  # НЕ ДЕЛАЙТЕ ТАК!
         token="abc123")    # НЕ ДЕЛАЙТЕ ТАК!
```

### ✅ Автоматическая защита:
```python
# Это будет автоматически очищено
log_json(logger, "Config loaded", 
         api_key="sk-...",  # Станет "***"
         model="gpt-4")     # Останется "gpt-4"
```

