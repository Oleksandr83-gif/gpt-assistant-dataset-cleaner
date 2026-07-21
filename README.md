# gpt-assistant-dataset-cleaner

GPT-powered dataset cleaner and reconstructor for multilingual assistant SFT (ru/ua/de/en).

Инструмент для очистки и реконструкции датасетов с помощью GPT:
- проверяет пары (user / assistant),
- слегка улучшает и переписывает ответы,
- делает мультиязычные варианты (ru, ua, de, en),
- может восстанавливать диалоги даже из «сырого» текста (статьи, логи, куски HTML и т.п.).

## Установка

### Вариант 1: через `pip` (локальная разработка)

```bash
git clone https://github.com/Oleksandr83-gif/gpt-assistant-dataset-cleaner.git
cd gpt-assistant-dataset-cleaner

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

Либо установить напрямую зависимости:

```bash
pip install -r requirements.txt
```

### Вариант 2: установка как пакет

После публикации на PyPI:

```bash
pip install gpt-assistant-dataset-cleaner
```

## Настройка

Скрипт читает ключ и модель из переменных окружения или `.env`:

- `OPENAI_API_KEY` — ключ OpenAI (обязательно),
- `GPT_MODEL` — имя модели (по умолчанию `gpt-4o-mini`).

Безопасный шаблон настроек находится в `.env.example`:

```bash
cp .env.example .env
# отредактируйте .env и вставьте ваш ключ
```

Можно также использовать `config.toml` (пока справочный):

```toml
[gpt]
model = "gpt-4o-mini"

[paths]
input_file  = "mix_0015.jsonl"
output_file = "assistant_sft_clean_mix0015.jsonl"
log_file    = "assistant_sft_rejected.jsonl"
state_file  = "assistant_sft_resume.json"
```

## Формат входа/выхода

### Входной JSONL

Ожидается файл вида `mix_0015.jsonl`, где **каждая строка — валидный JSON-объект**.
Скрипт поддерживает несколько форматов входа:

1. Уже готовые пары:

```json
{
  "domain": "dev",
  "messages": [
    {"role": "user", "content": "Как в Python прочитать JSON-файл?"},
    {"role": "assistant", "content": "Используй модуль `json`..."}
  ]
}
```

2. Простой формат без домена:

```json
{
  "user": "Как в Python прочитать JSON-файл?",
  "assistant": "Используй модуль `json`..."
}
```

3. Полный «сырой» текст:

```json
{
  "text": "Здесь может быть смесь лога, текста статьи, мусора и т.п."
}
```

Вариант (1) и (2) идут в лёгкую дообработку: GPT проверяет, чуть улучшает и делает вариации на разных языках.  
Вариант (3) идёт в более тяжёлую реконструкцию: GPT пытается вытащить 1–3 диалога и домены.

### Выходной JSONL

Выходной файл (по умолчанию `assistant_sft_clean_mix0015.jsonl`) содержит строки формата:

```json
{
  "domain": "dev",
  "lang": "ru",
  "messages": [
    {"role": "system", "content": "...system prompt..."},
    {"role": "user", "content": "Вопрос пользователя на русском"},
    {"role": "assistant", "content": "Ответ ассистента на русском"}
  ]
}
```

Для каждого исходного примера может быть несколько строк (варианты ru/ua/de/en).

## Запуск

После установки пакета через `pip install -e .` у вас должна появиться команда:

```bash
gpt-dataset-cleaner
```

По умолчанию она:

- читает входной файл `mix_0015.jsonl` из текущей директории,
- пишет результат в `assistant_sft_clean_mix0015.jsonl`,
- ведёт лог отклонённых строк в `assistant_sft_rejected.jsonl`,
- сохраняет прогресс в `assistant_sft_resume.json` (можно перезапускать с места остановки).

Также можно запустить напрямую:

```bash
python clener_gpt_only_oss.py
```

> Примечание: имя `clener_gpt_only_oss.py` сохранено для обратной совместимости с уже установленной CLI-командой. Переименование потребует отдельного релиза пакета.


## Лицензия

Код распространяется по лицензии **Apache 2.0** (см. файл `LICENSE`).  
Для удобства есть неофициальный перевод на русский в `LICENSE_RU.txt`.  
В случае расхождений юридически значимым является только английский текст.
