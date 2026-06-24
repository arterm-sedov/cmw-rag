# Архитектура развёртывания

## Хост: 10.9.7.7

### Запущенные сервисы

| Сервис | Порт | Процесс | Статус |
|--------|------|---------|--------|
| RAG Gradio UI | 7860 | `python rag_engine/api/app.py` | Работает |
| CMW-Mosec | 7998 | `cmw_mosec.v2.dynamic_server` | Работает |
| ChromaDB | 8000 | `chroma run --host 0.0.0.0 --port 8000` | Работает |

### Незапущенные сервисы

Отсутствуют — все основные сервисы активны.

---

## CMW-Mosec (порт 7998)

Один процесс Mosec обслуживает несколько моделей. Маршруты регистрируются в зависимости от переменных окружения.

Репозиторий: `github.com/arterm-sedov/cmw-mosec` — добавлен remote `cmw-team`.

### Маршруты

| Маршрут | Worker | Модель | Проверка |
|---------|--------|--------|----------|
| `POST /v1/embeddings` | EmbeddingWorkerV2 | `Qwen/Qwen3-Embedding-0.6B` | ✅ Возвращает эмбеддинги |
| `POST /v1/score` | ScoreWorkerV2 | `Qwen/Qwen3-Reranker-0.6B` | ✅ Возвращает оценки |
| `POST /v1/rerank` | RerankWorkerV2 | `Qwen/Qwen3-Reranker-0.6B` | ❌ Ошибка выполнения |
| `POST /v1/moderate` | GuardWorkerV2 | `Qwen/Qwen3Guard-Gen-0.6B` | ✅ Возвращает модерацию |

### Переменные окружения активных моделей

```
ACTIVE_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
ACTIVE_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
ACTIVE_GUARD_MODEL=Qwen/Qwen3Guard-Gen-0.6B
SERVER_PORT=7998
DEVICE=auto
DTYPE=float32
BATCH_SIZE=32
HF_TOKEN=<токен-huggingface>
```

### Запуск

```bash
cd cmw-mosec
source .venv/bin/activate
cmw-mosec serve
```

### Альтернативные модели

| Тип | Модели |
|-----|--------|
| Эмбеддинги | `ai-forever/FRIDA` (1536, ~4 ГБ), `Qwen/Qwen3-Embedding-0.6B` (1024, ~2 ГБ), `Qwen/Qwen3-Embedding-4B` (2560, ~12 ГБ), `Qwen/Qwen3-Embedding-8B` (4096, ~22 ГБ) |
| Ранжировщики | `DiTy/cross-encoder-russian-msmarco` (~2 ГБ), `BAAI/bge-reranker-v2-m3` (~2 ГБ), `Qwen/Qwen3-Reranker-0.6B` (~2 ГБ), `Qwen/Qwen3-Reranker-4B` (~12 ГБ), `Qwen/Qwen3-Reranker-8B` (~22 ГБ) |
| Модераторы | `Qwen/Qwen3Guard-Gen-0.6B` (~4 ГБ), `Qwen/Qwen3Guard-Gen-4B` (~10 ГБ), `Qwen/Qwen3Guard-Gen-8B` (~20 ГБ) |

---

## ChromaDB (порт 8000)

Векторное хранилище для поиска в RAG.

### Запуск

```bash
cd cmw-rag
source .venv/bin/activate
python rag_engine/scripts/start_chroma_server.py
```

### Активная коллекция

`mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768` (1024 измерения, соответствует Qwen3-Embedding-0.6B).

### Версионированные коллекции

| Коллекция | Содержимое |
|-----------|------------|
| `…_v5` | Документация платформы v5 |
| `…_v6` | Документация платформы v6 |

При смене модели эмбеддингов (иной размерности) требуется создать новую коллекцию и переиндексировать данные.

---

## RAG Gradio UI (порт 7860)

Точка входа: `rag_engine/api/app.py`.

Репозиторий: `github.com/arterm-sedov/cmw-rag` — добавлен remote `cmw-team`.

### Два агента

| Агент | Путь | Особенности |
|-------|------|-------------|
| Support Assistant | `/` | Полный интерфейс: выбор версии, панели метаданных, SRP (план решения), экспорт чата, MCP-инструменты (`ask_comindware`, `get_knowledge_base_articles`) |
| KB Assist | `/kb_assist` | Тот же агент, минимальный интерфейс (только чат и строка ввода, без панелей и выбора версии, `skip_srp=True`) |

Оба агента используют один и тот же обработчик LangChain (`chat_with_metadata`). KB Assist лишь скрывает панели метаданных и отключает SRP.

### Запуск

```bash
cd cmw-rag
source .venv/bin/activate
python rag_engine/api/app.py
```

### Схема связей

```
Пользователь (браузер)
    │
    ├──→ 10.9.7.7:7860/          (Support Agent)
    │
    ├──→ 10.9.7.7:7860/kb_assist (KB Assist Agent)
    │
    ├── Внешние интеграции (CMW Platform → RAG API):
    │       │
    │       ├──→ support.comindware.com
    │       │       POST /api/v1/cmw/process-support-request
    │       │       └── читает: systemSolution.Requests
    │       │       └── пишет:   systemSolution.agent_responses
    │       │
    │       └──→ lukoil.bau.cbap.ru
    │               POST /api/v1/cmw/summarize-document
    │               └── читает: ArchitectureManagement.Zaprosinarazrabotky
    │               └── пишет:  summary
    │
    └── Внутренние (RAG → подчинённые сервисы):
            │
            ├──→ localhost:7998  (Mosec — эмбеддинги, оценки, модерация)
            ├──→ localhost:8000  (ChromaDB — векторный поиск)
            └──→ api.openrouter.ai (LLM, инференс)
```

### Переменные окружения

```
# LLM
OPENROUTER_API_KEY=<ключ>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_LLM_PROVIDER=openrouter
DEFAULT_MODEL=deepseek/deepseek-v4-pro

# Эмбеддинги
EMBEDDING_PROVIDER_TYPE=mosec
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
MOSEC_EMBEDDING_ENDPOINT=http://localhost:7998/v1/embeddings

# Ранжирование
RERANK_ENABLED=true
RERANKER_PROVIDER_TYPE=mosec
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANKER_TIMEOUT=60.0
RERANKER_MAX_RETRIES=3
MOSEC_RERANKER_ENDPOINT=http://localhost:7998/v1/score

# Модерация
GUARD_ENABLED=true
GUARD_BLOCK_THRESHOLD=unsafe
GUARD_PROVIDER_TYPE=mosec
GUARD_MOSEC_ENDPOINT=http://localhost:7998/v1/moderate

# ChromaDB
CHROMADB_PORT=8000
CHROMA_CLIENT_HOST=localhost
CHROMA_SERVER_BIND=0.0.0.0
CHROMADB_COLLECTION=mkdocs_kb_qwen06b_linux_qwen_default_instruct_chunk_768
CHROMADB_PERSIST_DIR=~/cmw-rag/data/chromadb_data

# Параметры поиска
TOP_K_RETRIEVE=20
TOP_K_RERANK=10
CHUNK_SIZE=768
CHUNK_OVERLAP=75
RETRIEVAL_MULTIQUERY_ENABLED=true
RETRIEVAL_MULTIQUERY_MAX_SEGMENTS=4
RETRIEVAL_MULTIQUERY_SEGMENT_TOKENS=448
RETRIEVAL_QUERY_DECOMP_ENABLED=true
RETRIEVAL_QUERY_DECOMP_MAX_SUBQUERIES=4

# Gradio
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_LOCALE=ru

# Интеграция с CMW Platform — Primary (агент поддержки)
CMW_BASE_URL=https://support.comindware.com
CMW_LOGIN=<логин>
CMW_PASSWORD=<пароль>
CMW_TIMEOUT=30
CMW_API_KEY=<ключ-api>

# Интеграция с CMW Platform — Secondary / Лукойл (саммари)
CMW2_BASE_URL=<url-второй-платформы>
CMW2_LOGIN=<логин>
CMW2_PASSWORD=<пароль>
CMW2_TIMEOUT=30
CMW2_API_KEY=<ключ-api>

# Рассуждения LLM
LLM_REASONING_ENABLED=true
LLM_REASONING_MAX_TOKENS=1200
LLM_REASONING_EXCLUDE_FROM_RESPONSE=true
```

---

## Интеграция с CMW Platform

RAG взаимодействует с двумя экземплярами Comindware Platform через FastAPI и YAML-пайплайны. Оба работают по принципу fire-and-forget: прочитать запись — запустить агента в фоне — записать результат.

### Два экземпляра

| Экземпляр | Назначение | Файл конфигурации | Префикс env |
|-----------|-----------|-------------------|-------------|
| Primary | Поддержка: заявка → ответ RAG → запись ответа | `cmw_platform.yaml` | `CMW_` |
| Secondary (Лукойл) | Саммари документов: скачать → LLM → записать | `cmw_platform_secondary.yaml` | `CMW2_` |

### Primary: пайплайн обработки заявки

**Endpoint:** `POST /api/v1/cmw/process-support-request`

**Схема:** `{"request_id": "string"}`

**Процесс:**
```
CMW Platform (support.comindware.com)
    │ POST /api/v1/cmw/process-support-request {request_id}
    ▼
RAG FastAPI
    │ 1. Аутентификация по X-API-Key (если задан CMW_API_KEY)
    │ 2. Чтение записи из systemSolution.Requests
    │    Поля: name, Description, currentBuild, browserDetails
    │ 3. Сборка markdown-запроса по шаблону
    │ 4. Ответ 200: {success: true, message: "Request fetched, agent started at …"}
    │ 5. Фоновый поток → агент LangChain (ask_comindware_structured)
    │ 6. Преобразование ответа агента в выходные поля
    │ 7. Запись ответа в systemSolution.agent_responses
    │    (связь с исходной заявкой через support_request)
    ▼
Ответ записан в CMW Platform
```

**YAML-конфигурация пайплайна** (`cmw_platform.yaml`):

```yaml
pipeline:
  input:
    application: systemSolution
    template: Requests
    attributes:
      support_case_title: name
      support_case_question: Description
      product_version: currentBuild
      user_browser: browserDetails

  request_template: |
    ---
    - product version: {product_version}
    - user browser: {user_browser}
    ---
    # {support_case_title}
    {support_case_question}

  output:
    application: systemSolution
    template: agent_responses
    record_attribute: support_request
    linked_template: Requests
```

Сопоставление ответа агента с атрибутами (`mapping.py`):

- `answer` — окончательный ответ
- `question_for_agent` — исходный запрос с YAML-шапкой
- `agent_thinking` — результат исследования SGR
- `issue_area` — категория из `category_enum`
- `kb_articles` — процитированные статьи

### Secondary / Лукойл: саммари документов

**Endpoint:** `POST /api/v1/cmw/summarize-document`

**Схема:** `{"request_id": "string"}`

**Процесс:**
```
CMW Platform (lukoil.bau.cbap.ru, приложение: ArchitectureManagement)
    │ POST /api/v1/cmw/summarize-document {record_id}
    ▼
RAG FastAPI
    │ 1. Аутентификация по X-API-Key (если задан CMW2_API_KEY)
    │ 2. Чтение записи из ArchitectureManagement.Zaprosinarazrabotky
    │    Поля: Commerpredloshenie (документ), promt (запрос пользователя)
    │ 3. Загрузка документа → извлечение текста
    │ 4. Ответ 200: {success: true, message: "Начата обработка данных"}
    │ 5. Фоновый поток → create_summary_agent (LangChain)
    │    Агент с веб-поиском, системный промпт из YAML
    │ 6. Преобразование саммари в HTML
    │ 7. Запись саммари в атрибут summary записи
    ▼
Саммари записано в CMW Platform
```

**YAML-конфигурация** (`cmw_platform_secondary.yaml`):

```yaml
pipeline:
  input:
    application: ArchitectureManagement
    template: Zaprosinarazrabotky
    attributes:
      document_file: Commerpredloshenie
      user_prompt: promt

  output:
    application: ArchitectureManagement
    template: Zaprosinarazrabotky
    summary_attribute: summary
    summary_as_html: true

  system_prompt: |
    Ты — профессиональный бизнес-ассистент. Твоя задача — составлять
    краткие и информативные резюме деловых документов.
    …
```

### Pydantic-схемы

```python
class ProcessSupportRequest(BaseModel):
    request_id: str

class SummarizeDocumentRequest(BaseModel):
    request_id: str

class ProcessResult:
    success: bool
    message: str | None = None
    error: str | None = None

class HTTPResponse(BaseModel):
    success: bool
    status_code: int
    raw_response: dict | str | None = None
    error: str | None = None
    base_url: str

class APIResponse(BaseModel):
    response: Any | None = None
    success: bool | None = None
    error: str | None = None

class RequestConfig(BaseModel):
    base_url: str
    login: str
    password: str
    timeout: int = 30
```

### Category Enum

Категории обращений загружаются из `cmw_platform.yaml` → `category_enum`. Список получается из CMW Platform скриптом `scripts/fetch_issue_areas.py`. Примеры: `account`, `api`, `deployment`, `email`, `performance`, `integrations` и др.

---

## Синхронизация корпуса (MkDocs)

Синхронизирует документацию из внешнего MkDocs-репозитория в ChromaDB для индексации RAG.

### Systemd-таймер

| Компонент | Значение |
|-----------|----------|
| Таймер | `cmw-rag-corpus-sync.timer` — включён и активен |
| Расписание | Каждые 6 часов в 00:00, 06:00, 12:00, 18:00 UTC (+10 мин случайной задержки) |
| Сервис | `cmw-rag-corpus-sync.service` — oneshot, запускается по таймеру |
| Команда | `sync_mkdocs_corpus.py --index --corpus all` |

### Установка таймера

```bash
cp systemd/cmw-rag-corpus-sync.{service,timer} ~/.config/systemd/user/
systemctl --user enable --now cmw-rag-corpus-sync.timer
```

### Внешний репозиторий

| Параметр | Значение |
|----------|----------|
| Remote | `github.com/arterm-sedov/cbap-mkdocs-ru` |
| Ветка | `platform_v6` |
| Sparse path | `phpkb_content_rag` |
| Локальная копия | `.reference-repos/cbap-mkdocs-ru` |

### Версии корпуса

| Версия | Путь в репозитории | Суффикс коллекции ChromaDB |
|--------|--------------------|----------------------------|
| v5 | `phpkb_content_rag/798-platform_v5` | `_v5` |
| v6 | `phpkb_content_rag/896-platform_v6` | `_v6` |

Скрипт использует `git sparse-checkout` для загрузки только каталога `phpkb_content_rag`, затем индексирует новые и изменённые файлы. Если изменений нет — индексация пропускается.

### Ручной запуск

```bash
python rag_engine/scripts/sync_mkdocs_corpus.py --index --corpus all
python rag_engine/scripts/sync_mkdocs_corpus.py --index --corpus v6
```

---

## Инструкция по восстановлению

1. **Установить Python 3.11+** и создать виртуальное окружение для каждого проекта.
2. **Установить зависимости:** `pip install -e .` в cmw-mosec, `pip install -r rag_engine/requirements.txt` в cmw-rag.
3. **Установить ChromaDB:** `pip install chromadb`.
4. **Настроить `.env`:** в cmw-mosec (модели, порт, токен HuggingFace) и в cmw-rag (ключи LLM, учётные данные платформы).
5. **Запустить Mosec:** `cmw-mosec serve` — при первом запуске скачает модели (~8 ГБ видеопамяти GPU).
6. **Запустить ChromaDB:** `python rag_engine/scripts/start_chroma_server.py`.
7. **Построить индекс:** запустить синхронизацию корпуса для наполнения коллекций.
8. **Запустить RAG UI:** `python rag_engine/api/app.py`.
9. **Настроить webhook'и CMW Platform:**
   - `POST http://<host>:7860/api/v1/cmw/process-support-request` (support.comindware.com)
   - `POST http://<host>:7860/api/v1/cmw/summarize-document` (lukoil.bau.cbap.ru)
   - Оба принимают `{"request_id": "…"}` и опциональный заголовок `X-API-Key`.
10. **Проверить:**
    - `curl http://localhost:7998/v1/embeddings -X POST -d '{"input":"test","model":"Qwen/Qwen3-Embedding-0.6B"}'`
    - `curl http://localhost:7998/v1/moderate -X POST -d '{"input":"test"}'`
    - `curl http://localhost:8000/api/v1/heartbeat`
    - `curl http://localhost:7860/api/v1/cmw/process-support-request -X POST -d '{"request_id":"test"}'`
    - Открыть `http://localhost:7860` и `http://localhost:7860/kb_assist` в браузере.

### Требования к GPU

Mosec с тремя активными моделями требует около 8 ГБ видеопамяти GPU (эмбеддинг — 2 ГБ, ранжировщик — 2 ГБ, модератор — 4 ГБ). Все воркеры работают на одном устройстве через пакетную обработку Mosec.

---

## Соседние репозитории

### cmw-mosec

| Параметр | Значение |
|----------|----------|
| Репозиторий | `github.com/arterm-sedov/cmw-mosec` (добавлен remote `cmw-team`) |
| Точка входа | `cmw_mosec.cli:cli` (Click CLI) |
| Модуль сервера | `cmw_mosec.v2.dynamic_server` |
| Конфигурация моделей | `config/models.yaml` |
| Использование | Активно: эмбеддинги, оценки, модерация на этом хосте |

Запуск: `cmw-mosec serve` (читает `ACTIVE_*_MODEL`, запускается на `SERVER_PORT`).

### cmw-vllm

CLI-утилита для управления inference-серверами vLLM (совместимы с API OpenAI). Может использоваться как локальная замена OpenRouter.

| Параметр | Значение |
|----------|----------|
| Репозиторий | `github.com/arterm-sedov/cmw-vllm` (добавлен remote `cmw-team`) |
| Точка входа | `cmw_vllm.cli:cli` (Click CLI) |
| Реестр моделей | `cmw_vllm/model_registry.py` |
| Активная модель (если запущен) | `openai/gpt-oss-20b` |
| Порт по умолчанию | 8000 |
| Статус | Не развёрнут. Порт 8000 занят ChromaDB. |

```bash
# .env: VLLM_MODEL=openai/gpt-oss-20b  VLLM_PORT=8000  VLLM_HOST=0.0.0.0
cmw-vllm start openai/gpt-oss-20b
```

RAG подключается через `VLLM_BASE_URL=http://<host>:8000/v1` и `DEFAULT_LLM_PROVIDER=vllm`.

---

## Заметки

- Маршрут `/v1/rerank` в Mosec зарегистрирован, но возвращает ошибку выполнения. RAG использует `/v1/score`, который работает корректно.
- vLLM может служить локальным LLM-бэкендом вместо OpenRouter при развёртывании на отдельном хосте со свободным GPU и портом.
- При смене модели эмбеддингов (иной размерности) следует создать новую коллекцию ChromaDB и переиндексировать данные.
