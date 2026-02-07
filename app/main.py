import os
import re
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)

app = FastAPI(title="SEO Agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

class AnalyzeResponse(BaseModel):
    task_id: str

TasksStore: Dict[str, Dict[str, Any]] = {}

PLAN_SYSTEM_PROMPT = (
    "Ты эксперт по SEO-аналитике. Составь пошаговый план (5-6 шагов) для анализа сайта по ссылке и генерации SEO-ядра. "
    "Верни СТРОГО валидный JSON без пояснений. Формат: {\"steps\": [{\"title\": str, \"prompt\": str}, ...]}"
)

FINAL_SYSTEM_PROMPT = (
    "Ты эксперт по SEO. На основе промежуточных результатов шагов собери полноценное SEO-ядро. "
    "Верни СТРОГО валидный JSON с ключами: keywords (список), clusters (список объектов с name и items), "
    "meta (объект с description, h1, title), notes (строка)."
)

SAFE_TEXT_MAX = 150_000  # chars


def add_log(task_id: str, message: str) -> None:
    TasksStore[task_id]["logs"].append(message)


def extract_json(text: str) -> Dict[str, Any]:
    """Попытка извлечь JSON из ответа модели."""
    # Сначала пробуем как есть
    try:
        return json.loads(text)
    except Exception:
        pass
    # Ищем самый большой JSON-объект
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # Пустой фоллбек
    return {}


def clean_html(html: str) -> str:
    """Удаляет теги script/style и сжимает пробелы, возвращает текст страницы."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Сжатие пробелов/переводов строк
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > SAFE_TEXT_MAX:
            text = text[:SAFE_TEXT_MAX]
        return text
    except Exception:
        # В случае проблем вернём усечённый сырой html как fallback
        raw = html
        if len(raw) > SAFE_TEXT_MAX:
            raw = raw[:SAFE_TEXT_MAX]
        return raw


async def fetch_page_text(url: str) -> Optional[str]:
    try:
        timeout = httpx.Timeout(15.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as ac:
            resp = await ac.get(url)
            resp.raise_for_status()
            return clean_html(resp.text)
    except Exception:
        return None


async def openai_chat(messages: List[Dict[str, str]], response_format_json: bool = False) -> str:
    kwargs = {"model": OPENAI_MODEL, "messages": messages}
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def run_analysis_task(task_id: str, url: str) -> None:
    try:
        add_log(task_id, "Запуск задачи анализа…")
        page_text = await fetch_page_text(url)
        if page_text:
            add_log(task_id, f"Текст страницы получен ({len(page_text)} символов)")
        else:
            add_log(task_id, "Не удалось получить HTML, используем только ссылку")

        # 1) Получаем план шагов
        add_log(task_id, "Запрашиваем план шагов у модели…")
        plan_text = await openai_chat(
            [
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"url": url}, ensure_ascii=False)},
            ],
            response_format_json=True,
        )
        plan_obj = extract_json(plan_text)
        steps = plan_obj.get("steps") if isinstance(plan_obj, dict) else None
        if not steps or not isinstance(steps, list):
            raise RuntimeError("Модель вернула некорректный план. Повторите позже.")
        add_log(task_id, f"Получен план из {len(steps)} шагов")

        # 2) Выполняем шаги
        step_results: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            title = (step.get("title") or f"Шаг {idx}").strip()
            prompt = (step.get("prompt") or "").strip()
            add_log(task_id, f"Выполняем: {title}")

            user_payload = {
                "source_url": url,
                "page_text_excerpt": page_text or "",
                "instruction": prompt,
            }
            step_answer = await openai_chat(
                [
                    {"role": "system", "content": "Ты помощник по технич. SEO- и контент-анализу."},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ]
            )
            step_results.append({"title": title, "output": step_answer})
            add_log(task_id, f"Готово: {title}")

        # 3) Финальный анализ
        add_log(task_id, "Формируем финальный SEO-результат…")
        final_payload = {
            "source_url": url,
            "page_text_excerpt": page_text or "",
            "intermediate_results": step_results,
        }
        final_text = await openai_chat(
            [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(final_payload, ensure_ascii=False)},
            ],
            response_format_json=True,
        )
        final_obj = extract_json(final_text)
        TasksStore[task_id]["result"] = final_obj or {"raw": final_text}
        TasksStore[task_id]["done"] = True
        add_log(task_id, "Задача завершена")

    except Exception as e:
        TasksStore[task_id]["error"] = str(e)
        TasksStore[task_id]["done"] = True
        add_log(task_id, f"Ошибка: {e}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(url: str = Form(...)):
    task_id = str(uuid.uuid4())
    TasksStore[task_id] = {"logs": [], "done": False, "result": None, "error": None}
    asyncio.create_task(run_analysis_task(task_id, url))
    return AnalyzeResponse(task_id=task_id)


@app.get("/status/{task_id}")
async def status(task_id: str):
    data = TasksStore.get(task_id)
    if not data:
        return JSONResponse(status_code=404, content={"error": "task not found"})
    return data
