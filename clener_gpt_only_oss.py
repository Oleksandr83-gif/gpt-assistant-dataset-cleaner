# Copyright 2025 Oleksandr Matvieiev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset cleaner + GPT reconstructor (multilingual SFT assistant format)
Клинер датасетов + GPT-реконструктор (мульти-язычный SFT-формат ассистента)

- Берёт сырые JSONL-файлы (инструкции, диалоги, тексты, мусор).
- Превращает в аккуратный формат: domain + lang + messages (system/user/assistant).
- Дублирует пары на нескольких языках: ru / ua / de / en.
- Умеет:
  * дообрабатывать уже готовые пары (domain + messages),
  * реконструировать диалоги из "сырого" текста через GPT.

RU/EN comments are duplicated; Russian + English side by side.
"""

import json
import os
import re
import time
import hashlib
import sys
import unicodedata
from typing import Optional, Dict, Any, List

import requests
from tqdm import tqdm

# =========================
# БАЗОВЫЕ ПУТИ / BASIC PATHS
# =========================

# Root directory of this script
# Корневая папка для этого скрипта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input JSONL with raw / mixed data
# Входной JSONL с сырыми / смешанными данными
INPUT_FILE = os.path.join(
    BASE_DIR,
    "mix_0015.jsonl"  # <-- change if your file has another name / поменяй, если имя другое
)

# Output JSONL with cleaned assistant SFT data
# Выходной JSONL с очищенными парами для обучения ассистента
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "assistant_sft_clean_mix0015.jsonl"
)

# Log of rejected / problematic examples
# Лог отклонённых / проблемных примеров
LOG_FILE = os.path.join(
    BASE_DIR,
    "assistant_sft_rejected.jsonl"
)

# State file for resume
# Файл состояния для возобновления обработки
STATE_FILE = os.path.join(BASE_DIR, "assistant_sft_resume.json")

# =========================
# DOMAINS & SYSTEM PROMPTS
# =========================

# List of supported assistant domains
# Список поддерживаемых доменов ассистента
ASSISTANT_DOMAINS = {
    "general",
    "dev",
    "law",
    "police",
    "med",
    "driver",
    "pilot",
    "car_tech",
    "pc_tech",
    "strategy",
    "business",
    "home",
    "edu",
    "safety",
    "meta",
    "reserved",
}

# Default domain if heuristic can't decide
# Домен по умолчанию, если эвристика не смогла выбрать
DEFAULT_DOMAIN = "general"

# System prompt templates per domain
# Шаблоны системных промптов для каждого домена
SYSTEM_PROMPTS: Dict[str, str] = {
    "general": (
        "Ты — универсальный ассистент для человека.\n"
        "Отвечай доброжелательно, логично и по существу, без лишней воды.\n"
        "You are a general-purpose assistant.\n"
        "Answer kindly, logically and to the point, without unnecessary fluff."
    ),
    "dev": (
        "Ты — ассистент разработчика.\n"
        "Помогаешь писать и объяснять код, находить ошибки и проектировать архитектуру.\n"
        "Отвечай чётко, структурировано, по возможности с примерами кода.\n"
        "You are a developer assistant.\n"
        "Help with code, debugging and architecture. Be clear, structured, with code examples where possible."
    ),
    "law": (
        "Ты — ассистент-юрист.\n"
        "Объясняй правовые вопросы простым языком, ссылайся на нормы права, когда это уместно.\n"
        "Не давай советов, нарушающих закон, и не утверждай виновность людей в спорных делах.\n"
        "You are a legal assistant.\n"
        "Explain legal topics in simple language, refer to law where appropriate.\n"
        "Do not give advice that breaks the law and do not claim people are guilty in disputed cases."
    ),
    "police": (
        "Ты — ассистент для полицейской и криминальной аналитики.\n"
        "Анализируй ситуации с точки зрения закона и безопасности, избегай домыслов.\n"
        "You are an assistant for police / crime analysis.\n"
        "Analyze situations in terms of law and safety, avoid speculation."
    ),
    "med": (
        "Ты — медицинский ассистент.\n"
        "Объясняй медицинские темы, но не ставь диагнозы и не назначай лечение.\n"
        "Всегда советуй обратиться к врачу за очной консультацией.\n"
        "You are a medical assistant.\n"
        "Explain medical topics, but do not diagnose or prescribe treatment.\n"
        "Always recommend seeing a doctor in person."
    ),
    "driver": (
        "Ты — ассистент водителя.\n"
        "Помогаешь с вопросами вождения, ПДД, логистики и маршрутов.\n"
        "You are a driver assistant.\n"
        "Help with driving questions, traffic rules and route planning."
    ),
    "pilot": (
        "Ты — ассистент пилота.\n"
        "Обсуждай авиацию и процедуры на высоком уровне, без инструкций по опасным действиям.\n"
        "You are a pilot assistant.\n"
        "Discuss aviation and procedures at a high level, no step-by-step dangerous instructions."
    ),
    "car_tech": (
        "Ты — ассистент по ремонту и диагностике автомобилей.\n"
        "Помогаешь разбираться с неисправностями, но всегда напоминаешь о безопасности.\n"
        "You are a car repair assistant.\n"
        "Help with car diagnostics, but always remind about safety."
    ),
    "pc_tech": (
        "Ты — ассистент по ремонту компьютеров и электроники.\n"
        "Помогаешь диагностировать проблемы, объясняй шаги аккуратно и последовательно.\n"
        "You are a PC/electronics repair assistant.\n"
        "Help diagnose issues and explain steps carefully and sequentially."
    ),
    "strategy": (
        "Ты — ассистент по стратегиям и планированию.\n"
        "Помогаешь продумывать планы, оценивать риски и варианты действий.\n"
        "You are a strategy and planning assistant.\n"
        "Help design plans, evaluate risks and options."
    ),
    "business": (
        "Ты — ассистент по бизнесу и управлению.\n"
        "Разбираешься в процессах, финансах и менеджменте, объясняй без лишнего жаргона.\n"
        "You are a business and management assistant.\n"
        "Understand processes, finance and management; explain without unnecessary jargon."
    ),
    "home": (
        "Ты — ассистент по быту и семье.\n"
        "Помогаешь с повседневными задачами, организацией дома и воспитанием детей.\n"
        "You are a home & family assistant.\n"
        "Help with daily tasks, home organization and kids."
    ),
    "edu": (
        "Ты — образовательный ассистент.\n"
        "Объясняй сложные вещи простым языком, шаг за шагом.\n"
        "You are an educational assistant.\n"
        "Explain complex things in simple language, step by step."
    ),
    "safety": (
        "Ты — ассистент по безопасности и комплаенсу.\n"
        "Помогаешь оценивать риски, соблюдение закона и правил.\n"
        "You are a safety & compliance assistant.\n"
        "Help evaluate risks, legal compliance and rule following."
    ),
    "meta": (
        "Ты — мета-ассистент.\n"
        "Можешь обсуждать протоколы, архитектуру модели и правила её работы.\n"
        "You are a meta-assistant.\n"
        "You can discuss protocols, model architecture and its rules."
    ),
    "reserved": (
        "Ты — ассистент специального назначения.\n"
        "Этот домен зарезервирован для будущих режимов.\n"
        "You are a special-purpose assistant.\n"
        "This domain is reserved for future modes."
    ),
}

# =========================
# РЕЗЮМЕ СОСТОЯНИЯ / RESUME STATE
# =========================

def load_state(path: str) -> dict:
    """
    Load processing state from JSON file (offset + line count).
    Загрузка состояния обработки из JSON (offset + количество строк).
    """
    if not os.path.exists(path):
        return {"offset": 0, "lines": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        return {"offset": int(s.get("offset", 0)), "lines": int(s.get("lines", 0))}
    except Exception:
        return {"offset": 0, "lines": 0}


def save_state_atomic(path: str, offset: int, lines: int):
    """
    Atomically save processing state.
    Атомарно сохраняет состояние обработки.
    """
    tmp = path + ".tmp"
    data = {"offset": int(offset), "lines": int(lines), "ts": time.time()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

# =========================
# НАСТРОЙКИ GPT / GPT SETTINGS
# =========================

GPT_API_URL = "https://api.openai.com/v1/chat/completions"

# Model name is configurable via env var GPT_MODEL
# Имя модели задаётся через переменную окружения GPT_MODEL
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# API key only from environment variable (no hard-coded secrets!)
# API-ключ берётся только из переменной окружения (никаких хардкодов!)
GPT_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Length limits for input / output filtering
# Ограничения по длине для фильтрации входа / выхода
MAX_INPUT_CHARS = 6000
MIN_TEXT_CHARS = 20
MIN_PAIR_CHARS = 10
MAX_PAIR_CHARS = 2048
DEBUG_HTTP = True

# Target languages for duplication
# Целевые языки для дублирования пары
TARGET_LANGS = ["ru", "ua", "de", "en"]  # Russian / Ukrainian / German / English

# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ / HELPER FUNCTIONS
# =========================

def hash_pair(user: str, assistant: str, lang: str = "") -> str:
    """
    Stable hash for (user, assistant, lang) to detect duplicates.
    Стабильный хеш для (user, assistant, lang) для поиска дубликатов.
    """
    m = hashlib.md5()
    m.update((lang + "\n" + user + "\n" + assistant).encode("utf-8"))
    return m.hexdigest()


def safe_json_loads(s: str) -> Optional[dict]:
    """
    Safe JSON parse: return None on error.
    Безопасный json.loads: при ошибке возвращает None.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Try to extract JSON dict from raw model text.
    Пытается вытащить JSON-объект из произвольного текста модели.
    """
    obj = safe_json_loads(text)
    if isinstance(obj, dict):
        return obj

    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        obj = safe_json_loads(m.group(1))
        if isinstance(obj, dict):
            return obj

    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        obj = safe_json_loads(m.group(1))
        if isinstance(obj, dict):
            return obj
    return None


def debug_http(resp: Optional[requests.Response]):
    """
    Debug-print HTTP errors from OpenAI.
    Отладочный вывод HTTP-ошибок от OpenAI.
    """
    if not DEBUG_HTTP or resp is None:
        return
    if resp.status_code == 200:
        return
    print(f"\nHTTP {resp.status_code}:")
    txt = resp.text
    print(txt[:1000])
    print("-----\n")

# =========================
# НОРМАЛИЗАЦИЯ ТЕКСТА / TEXT NORMALIZATION
# =========================

HTML_TAG_PATTERN = re.compile(
    r"</?(?:div|span|p|br|a|strong|em|b|i|h[1-6]|ul|ol|li|code|pre|img|table|tr|td|th|tbody|thead|tfoot|header|footer|section|article|nav|style|script)[^>]*>",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(r"https?://\S+")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def normalize_text(text: str) -> str:
    """
    Normalize text: NFC, strip HTML, filter control chars, normalize spacing.
    Нормализует текст: NFC, вырезает HTML, убирает управляющие символы, чистит пробелы.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\t", " ")
    text = HTML_TAG_PATTERN.sub("", text)

    def _url_repl(m: re.Match) -> str:
        url = m.group(0)
        if len(url) > 200:
            return "[URL]"
        return url

    text = URL_PATTERN.sub(_url_repl, text)
    text = EMAIL_PATTERN.sub("[EMAIL]", text)

    cleaned_chars = []
    for ch in text:
        if ch == "\n":
            cleaned_chars.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            continue
        cleaned_chars.append(ch)
    text = "".join(cleaned_chars)
    text = text.replace("<mask>", "")

    lines = text.split("\n")
    lines = [ln.rstrip(" ") for ln in lines]
    text = "\n".join(lines).strip()
    return text

# =========================
# ДОСТАЁМ СЫРОЙ ТЕКСТ ИЗ JSON / EXTRACT RAW TEXT
# =========================

def extract_raw_text(obj: Dict[str, Any]) -> Optional[str]:
    """
    Extract raw text from various JSON formats (instruction, prompt, text fields).
    Достаёт текст из разных форматов JSON (instruction, prompt, text и т.п.).
    """
    # Already assistant format – handled elsewhere
    # Уже формат ассистента — здесь не обрабатываем
    if "domain" in obj and "messages" in obj and isinstance(obj["messages"], list):
        return None

    if "messages" in obj and isinstance(obj["messages"], list):
        return None

    if "instruction" in obj and "output" in obj:
        instr = str(obj["instruction"]).strip()
        out = str(obj["output"]).strip()
        if instr and out:
            return f"<|user|>\n{instr}\n<|assistant|>\n{out}"

    if "prompt" in obj and "completion" in obj:
        p = str(obj["prompt"]).strip()
        c = str(obj["completion"]).strip()
        if p and c:
            return f"<|user|>\n{p}\n<|assistant|>\n{c}"

    for k in ("text", "content", "raw_text"):
        if k in obj and isinstance(obj[k], str):
            t = obj[k].strip()
            if t:
                return t

    return None

# =========================
# GPT ENRICH (MULTILANG) FOR READY PAIRS
# GPT ОБРАБОТКА ГОТОВЫХ ПАР (МУЛЬТИЯЗЫК)
# =========================

def gpt_enrich_multilang(user: str,
                         assistant: str,
                         domain: str,
                         target_langs: List[str],
                         max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """
    GPT:
    - validates a user/assistant pair,
    - slightly enriches assistant answer,
    - generates variants in several languages (ru/ua/de/en).

    GPT:
    - проверяет пару (user/assistant),
    - немного расширяет ответ ассистента,
    - генерирует варианты на нескольких языках (ru/ua/de/en).

    Returns / Возвращает:
    {
      "accept": true/false,
      "variants": [
        {"lang": "ru", "user": "...", "assistant": "..."},
        ...
      ]
    }
    """
    if not GPT_API_KEY:
        print("❌ OPENAI_API_KEY не задан — GPT недоступен. / OPENAI_API_KEY is not set — GPT unavailable.")
        return None

    # Extra safety hints for sensitive domains
    # Дополнительные подсказки по безопасности для чувствительных доменов
    safety_hint = ""
    if domain in ("law", "police", "safety"):
        safety_hint = (
            "⚠ В юридических/правоохранительных вопросах НЕ утверждай виновность "
            "конкретных людей в спорных делах. Говори нейтрально: "
            "«по сообщениям СМИ», «есть обвинения, но итог неизвестен» и т.п.\n"
            "⚠ For legal / law-enforcement topics do NOT state that specific people "
            "are guilty in disputed cases. Use neutral phrasing.\n"
        )
    if domain == "med":
        safety_hint += (
            "⚠ В медицине НЕ ставь диагнозы и НЕ назначай лечение. "
            "Всегда добавляй, что нужен очный врач.\n"
            "⚠ In medical topics do NOT diagnose or prescribe treatment. "
            "Always add that an in-person doctor visit is required.\n"
        )

    system_prompt = (
        "Ты — редактор и переводчик датасета для обучения мульти-язычной модели ассистента.\n"
        "You are an editor and translator for a dataset to train a multilingual assistant model.\n\n"
        "На вход даётся пара (user, assistant). Твоя задача:\n"
        "You get a (user, assistant) pair. Your task:\n"
        "1) Проверить, что это осмысленный, пригодный пример (без спама и мусора).\n"
        "   Check it is meaningful and usable (no spam/noise).\n"
        "2) При необходимости немного расширить ответ assistant: добавить пару деталей, "
        "списки, примеры, но НЕ менять смысл и НЕ придумывать факты.\n"
        "   Optionally slightly expand assistant's answer (lists, examples) "
        "   without changing meaning or inventing facts.\n"
        "3) Для каждого языка из списка сделай естественный вариант этой пары на этом языке.\n"
        "   For each language in the list create a natural variant in that language.\n"
        "4) Можно использовать эмодзи (0–2 штуки) в ответах там, где это уместно, "
        "и списки с '-' или нумерацией для структурирования.\n"
        "   You may use 0–2 emojis and bullet/numbered lists to structure the answer.\n"
        f"{safety_hint}\n"
        "Формат ответа — СТРОГО JSON без пояснений и текста вокруг, вот такой структуры:\n"
        "Response format MUST be pure JSON with this structure, nothing else:\n"
        "{\n"
        "  \"accept\": true/false,\n"
        "  \"variants\": [\n"
        "    {\"lang\": \"ru\", \"user\": \"...\", \"assistant\": \"...\"},\n"
        "    {\"lang\": \"ua\", \"user\": \"...\", \"assistant\": \"...\"},\n"
        "    {\"lang\": \"de\", \"user\": \"...\", \"assistant\": \"...\"},\n"
        "    {\"lang\": \"en\", \"user\": \"...\", \"assistant\": \"...\"}\n"
        "  ]\n"
        "}\n"
        "Если для какого-то языка естественный перевод невозможен, просто пропусти этот язык.\n"
        "If some language is not natural for this pair, just skip that language.\n"
    )

    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"user": user, "assistant": assistant, "domain": domain,
                     "target_langs": target_langs},
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": 800,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT_API_KEY}",
    }

    for attempt in range(1, max_retries + 1):
        resp = None
        try:
            resp = requests.post(GPT_API_URL, json=payload, headers=headers, timeout=90)
            code = resp.status_code

            if code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                obj = extract_json_from_text(content)
                if not obj or not isinstance(obj, dict):
                    time.sleep(min(5, 2 ** attempt))
                    continue

                if not obj.get("accept", False):
                    return {"accept": False, "variants": []}

                variants = obj.get("variants", [])
                if not isinstance(variants, list) or not variants:
                    return {"accept": False, "variants": []}

                clean_vars = []
                for v in variants:
                    lang = str(v.get("lang", "")).lower()
                    if lang not in target_langs:
                        continue
                    u = str(v.get("user", "")).strip()
                    a = str(v.get("assistant", "")).strip()
                    if not u or not a:
                        continue
                    clean_vars.append({"lang": lang, "user": u, "assistant": a})

                if not clean_vars:
                    return {"accept": False, "variants": []}

                return {"accept": True, "variants": clean_vars}

            if code in (401, 403):
                debug_http(resp)
                print("❌ GPT авторизация сломалась. / GPT auth error.")
                return None

            if code == 404:
                debug_http(resp)
                print("❌ GPT: модель не найдена. / GPT model not found.")
                return None

            if code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(1, int(retry_after))
                    except ValueError:
                        wait = min(30, 2 ** attempt)
                else:
                    wait = min(30, 2 ** attempt)
                print(f"⏳ GPT retry {attempt}/{max_retries} через {wait}s (HTTP {code}) / retry in {wait}s")
                time.sleep(wait)
                continue

            debug_http(resp)
            return None

        except Exception as e:
            print(f"⚠️ GPT ошибка (попытка {attempt}/{max_retries}): {e} / GPT error (attempt {attempt}/{max_retries})")
            time.sleep(min(10, 2 ** attempt))

    return None

# =========================
# GPT RECONSTRUCT FROM RAW
# РЕКОНСТРУКЦИЯ ДИАЛОГОВ ИЗ СЫРОГО ТЕКСТА
# =========================

def gpt_reconstruct_from_raw(raw_text: str,
                             target_langs: List[str],
                             max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """
    GPT:
    - infers topic from raw text,
    - creates 1–3 user/assistant dialogues,
    - assigns domain per dialogue,
    - generates multilingual variants.

    GPT:
    - понимает тему куска текста,
    - создаёт 1–3 диалога user/assistant,
    - каждому диалогу задаёт domain,
    - делает мультиязычные варианты.
    """
    if not GPT_API_KEY:
        print("❌ OPENAI_API_KEY не задан — GPT недоступен. / OPENAI_API_KEY is not set — GPT unavailable.")
        return None

    system_prompt = (
        "Ты помогаешь готовить датасет для обучения мульти-язычной модели ассистента.\n"
        "You help to prepare a dataset for training a multilingual assistant model.\n\n"
        "На вход ты получаешь ОДИН фрагмент сырых данных: диалог, описание, новость, "
        "кусок статьи, лог, или смесь мусора.\n"
        "You receive ONE raw fragment: chat, description, news, article snippet, log, or mixed noise.\n\n"
        "Твоя задача:\n"
        "Your task:\n"
        "1) Понять общую тему фрагмента (примерно).\n"
        "   Roughly infer the main topic.\n"
        "2) Сконструировать на основе этой темы 1–3 осмысленных диалога "
        "(реплика пользователя и ответ ассистента).\n"
        "   Build 1–3 meaningful user/assistant dialogues based on that topic.\n"
        "   - Диалоги должны быть универсальными и безопасными.\n"
        "     Dialogues must be universal and safe.\n"
        "   - Не приписывай конкретным реальным людям преступления или факты, "
        "которых нет в тексте.\n"
        "     Do not attribute crimes or facts to real people that are not in the text.\n"
        "   - Убирай HTML, логи, технический мусор.\n"
        "     Remove HTML, logs and technical noise.\n"
        "3) Для каждого диалога выбери один домен модели ассистента из списка:\n"
        "   For each dialogue choose one assistant domain from this list:\n"
        f"   {sorted(ASSISTANT_DOMAINS)}\n"
        "4) Для каждого диалога сделай несколько языковых вариантов (ru, ua, de, en).\n"
        "   For each dialogue create several language variants (ru, ua, de, en).\n"
        "   - Вариант lang='ru' — и user, и assistant на русском.\n"
        "     For lang='ru' both user and assistant are in Russian.\n"
        "   - Аналогично для ua, de, en.\n"
        "   - Можно использовать эмодзи (0–2) и списки для структурирования.\n"
        "     You may use 0–2 emojis and lists for structure.\n"
        "5) Если из фрагмента вообще невозможно сделать нормальный диалог, "
        "   верни accept=false.\n"
        "   If you really cannot produce usable dialogues, return accept=false.\n\n"
        "Формат ответа — строго JSON без пояснений вокруг:\n"
        "Response format MUST be pure JSON, no extra text:\n"
        "{\n"
        "  \"accept\": true/false,\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"domain\": \"general\",           // один из разрешённых доменов\n"
        "      \"variants\": [\n"
        "        {\"lang\": \"ru\", \"user\": \"...\", \"assistant\": \"...\"},\n"
        "        {\"lang\": \"ua\", \"user\": \"...\", \"assistant\": \"...\"},\n"
        "        {\"lang\": \"de\", \"user\": \"...\", \"assistant\": \"...\"},\n"
        "        {\"lang\": \"en\", \"user\": \"...\", \"assistant\": \"...\"}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Если accept=false, можешь не указывать items или оставить пустым списком.\n"
        "If accept=false you may omit items or keep it empty.\n"
    )

    user_content = json.dumps(
        {"raw_text": raw_text[:MAX_INPUT_CHARS], "target_langs": target_langs},
        ensure_ascii=False,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GPT_API_KEY}",
    }

    body = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.5,
        "max_tokens": 1600,
        "stream": False,
    }

    for attempt in range(1, max_retries + 1):
        resp = None
        try:
            resp = requests.post(GPT_API_URL, json=body, headers=headers, timeout=120)
            code = resp.status_code
            if code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                obj = extract_json_from_text(content)
                if not obj or not isinstance(obj, dict):
                    time.sleep(min(5, 2 ** attempt))
                    continue
                return obj

            if code in (401, 403):
                debug_http(resp)
                print("❌ GPT авторизация сломалась. / GPT auth error.")
                return None

            if code == 404:
                debug_http(resp)
                print("❌ GPT: модель не найдена. / GPT model not found.")
                return None

            if code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(1, int(retry_after))
                    except ValueError:
                        wait = min(30, 2 ** attempt)
                else:
                    wait = min(30, 2 ** attempt)
                print(f"⏳ GPT reconstruct retry {attempt}/{max_retries} через {wait}s (HTTP {code}) / retry in {wait}s")
                time.sleep(wait)
                continue

            debug_http(resp)
            return None

        except Exception as e:
            print(f"⚠️ GPT reconstruct ошибка (попытка {attempt}/{max_retries}): {e} / GPT reconstruct error")
            time.sleep(min(10, 2 ** attempt))

    return None

# =========================
# ЭВРИСТИКА ДОМЕНА / DOMAIN HEURISTICS
# =========================

def guess_domain(user: str, assistant: str) -> str:
    """
    Very simple keyword-based heuristic to guess domain.
    Простая эвристика по ключевым словам для определения домена.
    """
    text = (user + " " + assistant).lower()

    dev_keywords = [
        "код", "скрипт", "функция", "класс", "компилятор", "баг", "стек", "исключение",
        "python", "java", "kotlin", "c++", "c#", "javascript", "typescript",
        "def ", "class ", "public ", "private ", "console.log", "#include", "import ",
    ]
    if any(kw in text for kw in dev_keywords):
        return "dev"

    law_keywords = [
        "статья", "уголовн", "гражданск", "административн",
        "закон", "кодекс", "§", "иск", "договор", "контракт",
        "liability", "fine", "право", "jurisdiction",
    ]
    if any(kw in text for kw in law_keywords):
        return "law"

    police_keywords = [
        "полиция", "правоохран", "преступлен", "угроза",
        "полицейский", "crime", "criminal", "offense", "felony",
    ]
    if any(kw in text for kw in police_keywords):
        return "police"

    safety_keywords = [
        "безопасност", "risk", "риск", "комплаенс", "compliance",
        "опасно", "опасность", "нарушение", "violation",
    ]
    if any(kw in text for kw in safety_keywords):
        return "safety"

    med_keywords = [
        "симптом", "диагноз", "врач", "лечение", "таблетк",
        "болит", "температура", "давление", "анализ", "blood test",
    ]
    if any(kw in text for kw in med_keywords):
        return "med"

    car_keywords = [
        "двигател", "коробка", "кпп", "масло", "тормоз",
        "авто", "машин", "engine", "gearbox", "обслуживание авто",
    ]
    if any(kw in text for kw in car_keywords):
        return "car_tech"

    pc_keywords = [
        "материнск", "материнка", "bios", "uefi", "видеокарт",
        "gpu", "cpu", "оперативк", "ddr", "ssd", "hdd",
        "northbridge", "southbridge", "вентилятор", "кулер",
    ]
    if any(kw in text for kw in pc_keywords):
        return "pc_tech"

    biz_keywords = [
        "бизнес", "прибыль", "убыток", "маржа", "доход", "расход",
        "инвестиции", "стартап", "налог", "steuer", "budget", "cashflow",
    ]
    if any(kw in text for kw in biz_keywords):
        return "business"

    strategy_keywords = [
        "стратегия", "план", "roadmap", "тактика", "долгосрочн",
        "краткосрочн", "цель", "target", "okrs", "kpi",
    ]
    if any(kw in text for kw in strategy_keywords):
        return "strategy"

    home_keywords = [
        "уборка", "дом", "квартира", "ремонт дома", "семья",
        "дети", "готовка", "кухня", "хозяйство",
    ]
    if any(kw in text for kw in home_keywords):
        return "home"

    edu_keywords = [
        "как выучить", "объясни", "шаг за шагом", "репетитор",
        "учёба", "учеба", "курс", "lesson", "explain", "учиться", "студент",
    ]
    if any(kw in text for kw in edu_keywords):
        return "edu"

    driver_keywords = [
        "пдд", "штраф за скорость", "водитель", "парковка",
        "автошкола", "fahrschule", "führerschein",
    ]
    if any(kw in text for kw in driver_keywords):
        return "driver"

    pilot_keywords = [
        "самолёт", "самолет", "пилот", "cockpit", "flight level",
        "boeing", "airbus", "ifr", "vfr",
    ]
    if any(kw in text for kw in pilot_keywords):
        return "pilot"

    meta_keywords = [
        "модель ассистента", "tokenizer", "протокол", "архитектура модели",
        "rlhf", "fine-tune", "pretrain", "transformer",
    ]
    if any(kw in text for kw in meta_keywords):
        return "meta"

    return DEFAULT_DOMAIN if DEFAULT_DOMAIN in ASSISTANT_DOMAINS else "general"

# =========================
# ЗАГРУЖАЕМ ГОТОВЫЕ ПАРЫ / LOAD EXISTING PAIRS
# =========================

def load_existing_pairs(path: str) -> set:
    """
    Load already cleaned examples to avoid duplicates.
    Загружает уже очищенные примеры, чтобы не плодить дубликаты.
    """
    seen = set()
    if not os.path.exists(path):
        return seen

    print("📂 Читаю уже готовый чистый датасет (для резюме/дублей)... / Loading existing clean dataset...", end=" ", flush=True)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = safe_json_loads(line)
                if not obj:
                    continue
                msgs = obj.get("messages")
                if not isinstance(msgs, list):
                    continue

                user_msg = None
                assistant_msg = None
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    r = m.get("role")
                    if r == "user" and user_msg is None:
                        user_msg = m
                    elif r == "assistant" and assistant_msg is None:
                        assistant_msg = m
                    if user_msg is not None and assistant_msg is not None:
                        break

                if not user_msg or not assistant_msg:
                    continue

                u = str(user_msg.get("content", "")).strip()
                a = str(assistant_msg.get("content", "")).strip()
                lang = str(obj.get("lang", "")).strip().lower()
                if u and a:
                    seen.add(hash_pair(u, a, lang))
        print(f"готово, найдено {len(seen)} пар. / done, {len(seen)} pairs found.")
    except Exception as e:
        print(f"\n⚠️ Ошибка чтения чистого датасета: {e} / Error reading clean dataset: {e}")
    return seen

# =========================
# ПРОВЕРКА GPT / GPT HEALTHCHECK
# =========================

def check_gpt_alive() -> bool:
    """
    Quick healthcheck ping for GPT.
    Быстрый healthcheck для GPT.
    """
    if not GPT_API_KEY:
        print("❌ OPENAI_API_KEY не задан — GPT недоступен. / OPENAI_API_KEY is not set — GPT unavailable.")
        return False
    try:
        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": "Healthcheck. Answer exactly: OK"},
                {"role": "user", "content": "ping"},
            ],
            "temperature": 0.0,
            "max_tokens": 2,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GPT_API_KEY}",
        }
        resp = requests.post(GPT_API_URL, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            debug_http(resp)
            return False
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip().upper()
        return content == "OK"
    except Exception as e:
        print(f"❌ GPT healthcheck error: {e}")
        return False

# =========================
# MAIN / ОСНОВНАЯ ЛОГИКА
# =========================

def main():
    # Check input file
    # Проверяем наличие входного файла
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Входной файл не найден: {INPUT_FILE} / Input file not found.")
        sys.exit(1)

    print("🔍 Проверяю доступность GPT... / Checking GPT availability...")
    if not check_gpt_alive():
        print("❌ GPT API не работает — скрипт завершён. / GPT API not responding — exiting.")
        sys.exit(1)
    print("✅ GPT доступен, начинаю обработку. / GPT is available, starting processing.\n")

    existing_hashes = load_existing_pairs(OUTPUT_FILE)

    state = load_state(STATE_FILE)
    resume_offset = state["offset"]
    resume_lines = state["lines"]

    total_in_file = sum(1 for _ in open(INPUT_FILE, "rb"))
    print(f"📊 Строк во входном файле: {total_in_file} / Lines in input file: {total_in_file}")

    if resume_offset > 0:
        print(f"🔁 RESUME: продолжу с offset={resume_offset}, lines={resume_lines} / will resume from here.")

    total_lines = resume_lines
    total_written = 0
    parse_errors = 0
    short_long_dropped = 0
    noise_dropped = 0
    duplicates = 0
    gpt_enrich_failed = 0
    gpt_reconstruct_failed = 0
    reused_ready_pairs = 0
    reconstructed_from_raw = 0

    print("🚀 Старт полной GPT-обработки (Recon + Multilang → Assistant JSONL)! / Starting full GPT pipeline...")

    last_save_t = time.time()
    save_every_seconds = 5
    save_every_lines = 200

    with open(INPUT_FILE, "rb") as fin, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as fout, \
         open(LOG_FILE, "a", encoding="utf-8") as flog:

        if resume_offset > 0:
            fin.seek(resume_offset)

        pbar = tqdm(
            total=total_in_file,
            initial=resume_lines,
            desc="Обработка (GPT-only) / Processing (GPT-only)",
            unit="line",
            dynamic_ncols=True
        )

        try:
            while True:
                raw = fin.readline()
                if not raw:
                    break

                next_pos = fin.tell()
                total_lines += 1
                pbar.update(1)

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                obj = safe_json_loads(line)
                if not obj:
                    parse_errors += 1
                    flog.write(json.dumps({"reason": "json_error", "raw": line[:500]}, ensure_ascii=False) + "\n")
                    continue

                # --- Variant 1: already assistant format (domain + messages) ---
                # --- Вариант 1: уже формат ассистента (domain + messages) ---
                if "domain" in obj and "messages" in obj and isinstance(obj["messages"], list):
                    msgs = obj["messages"]
                    user_msg = None
                    assistant_msg = None
                    for m in msgs:
                        if not isinstance(m, dict):
                            continue
                        r = m.get("role")
                        if r == "user" and user_msg is None:
                            user_msg = m
                        elif r == "assistant" and assistant_msg is None:
                            assistant_msg = m
                        if user_msg is not None and assistant_msg is not None:
                            break

                    if not user_msg or not assistant_msg:
                        noise_dropped += 1
                        flog.write(json.dumps({"reason": "ready_no_pair", "obj": obj}, ensure_ascii=False) + "\n")
                        continue

                    user_text = normalize_text(str(user_msg.get("content", "")))
                    assistant_text = normalize_text(str(assistant_msg.get("content", "")))
                    if len(user_text) < MIN_PAIR_CHARS or len(assistant_text) < MIN_PAIR_CHARS:
                        short_long_dropped += 1
                        flog.write(json.dumps(
                            {"reason": "ready_pair_too_short", "user": user_text, "assistant": assistant_text},
                            ensure_ascii=False
                        ) + "\n")
                        continue

                    domain_raw = str(obj.get("domain", DEFAULT_DOMAIN))
                    domain = domain_raw if domain_raw in ASSISTANT_DOMAINS else guess_domain(user_text, assistant_text)

                    enrich = gpt_enrich_multilang(user_text, assistant_text, domain, TARGET_LANGS)
                    if not enrich or not enrich.get("accept"):
                        gpt_enrich_failed += 1
                        flog.write(json.dumps(
                            {"reason": "gpt_enrich_failed", "user": user_text[:200], "assistant": assistant_text[:200]},
                            ensure_ascii=False
                        ) + "\n")
                        continue

                    variants = enrich["variants"]
                    sys_prompt = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["general"])

                    for v in variants:
                        lang = v["lang"]
                        u = normalize_text(v["user"])
                        a = normalize_text(v["assistant"])
                        if not u or not a:
                            continue
                        h = hash_pair(u, a, lang)
                        if h in existing_hashes:
                            duplicates += 1
                            continue
                        existing_hashes.add(h)

                        record = {
                            "domain": domain,
                            "lang": lang,
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": u},
                                {"role": "assistant", "content": a},
                            ],
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_written += 1
                        reused_ready_pairs += 1

                # --- Variant 2: raw / unformatted JSON ---
                # --- Вариант 2: сырой / неформатированный JSON ---
                else:
                    raw_text = extract_raw_text(obj)
                    if not raw_text:
                        noise_dropped += 1
                        flog.write(json.dumps({"reason": "no_text", "obj": obj}, ensure_ascii=False) + "\n")
                        continue

                    raw_text = raw_text.strip()
                    if len(raw_text) < MIN_TEXT_CHARS or len(raw_text) > MAX_INPUT_CHARS:
                        short_long_dropped += 1
                        flog.write(json.dumps(
                            {"reason": "len_filter", "text": raw_text[:500]},
                            ensure_ascii=False
                        ) + "\n")
                        continue

                    gpt_res = gpt_reconstruct_from_raw(raw_text, TARGET_LANGS)
                    if not gpt_res:
                        gpt_reconstruct_failed += 1
                        flog.write(json.dumps(
                            {"reason": "gpt_reconstruct_failed", "text": raw_text[:500]},
                            ensure_ascii=False
                        ) + "\n")
                        continue

                    if not gpt_res.get("accept", False):
                        gpt_reconstruct_failed += 1
                        flog.write(json.dumps(
                            {"reason": "gpt_reconstruct_reject", "text": raw_text[:500], "obj": gpt_res},
                            ensure_ascii=False
                        ) + "\n")
                        continue

                    items = gpt_res.get("items", [])
                    if not isinstance(items, list) or not items:
                        gpt_reconstruct_failed += 1
                        flog.write(json.dumps(
                            {"reason": "gpt_reconstruct_empty_items", "text": raw_text[:500], "obj": gpt_res},
                            ensure_ascii=False
                        ) + "\n")
                        continue

                    for item in items:
                        domain = str(item.get("domain", DEFAULT_DOMAIN)).strip()
                        if domain not in ASSISTANT_DOMAINS:
                            domain = DEFAULT_DOMAIN

                        variants = item.get("variants", [])
                        if not isinstance(variants, list):
                            continue

                        sys_prompt = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["general"])

                        for v in variants:
                            lang = str(v.get("lang", "")).lower()
                            if lang not in TARGET_LANGS:
                                continue
                            u = normalize_text(str(v.get("user", "")))
                            a = normalize_text(str(v.get("assistant", "")))
                            if not u or not a:
                                continue

                            h = hash_pair(u, a, lang)
                            if h in existing_hashes:
                                duplicates += 1
                                continue
                            existing_hashes.add(h)

                            record = {
                                "domain": domain,
                                "lang": lang,
                                "messages": [
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": u},
                                    {"role": "assistant", "content": a},
                                ],
                            }
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_written += 1
                            reconstructed_from_raw += 1

                now = time.time()
                if (now - last_save_t) >= save_every_seconds or (total_lines % save_every_lines == 0):
                    save_state_atomic(STATE_FILE, offset=next_pos, lines=total_lines)
                    last_save_t = now

        except KeyboardInterrupt:
            save_state_atomic(STATE_FILE, offset=fin.tell(), lines=total_lines)
            pbar.close()
            print("\n🛑 Остановлено пользователем. Прогресс сохранён. / Stopped by user, progress saved.")
            return

        pbar.close()

    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    print("\n✅ Готово (GPT-only). / Done (GPT-only).")
    print(f"Всего строк входа: {total_lines} / Total input lines: {total_lines}")
    print(f"Записано чистых примеров ассистента (всех языков): {total_written} / Clean examples written: {total_written}")
    print(f"Из них реконструировано из сырого текста: {reconstructed_from_raw} / Reconstructed from raw: {reconstructed_from_raw}")
    print(f"Повторно использовано готовых примеров: {reused_ready_pairs} / Reused ready pairs: {reused_ready_pairs}")
    print(f"Дубликаты: {duplicates}, короткие/длинные: {short_long_dropped} / duplicates, short/long filtered")
    print(f"Мусор/отклонено: {noise_dropped} / noise/rejected: {noise_dropped}")
    print(f"GPT enrich fail (готовые пары): {gpt_enrich_failed}")
    print(f"GPT reconstruct fail (сырые куски): {gpt_reconstruct_failed}")


if __name__ == "__main__":
    main()
