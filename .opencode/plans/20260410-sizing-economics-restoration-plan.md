# Plan: Restore Missing Content - Execution Log

## Step1: Add Token Sizing Tables

**Goal:** Rename H2 and add token calculation tables

**Location:** After line 352 (after "FinOps и юнит-экономика нагрузки" section)

**Changes:**
1. Rename H2 `## Экономический каркас и токен-экономика` → `## Модель затрат`
2. Add H3 `### Калькуляция по классам задач`
3. Add table "Класс агента (ориентир длины системного промпта)"
4. Add table "Класс данных по длине пользовательского текста"

**Source content (from original lines 489-509):**

```markdown
### Калькуляция по классам задач {: #sizing_token_calculation_classes }

Для оценки токенов при планировании бюджета используйте классификаторы по агентам и данным.

#### Класс агента (ориентир длины системного промпта) {: #sizing_agent_class_system_prompt_length }

!!! tip "Базовый тариф расчёта"

    Расчёт выполнен при медианном тарифе стандарт-сегмента (~300 ₽/млн токенов, вход=выход, типично для YandexGPT Lite/GLM-5/MiniMax-M2.7). Для точного подсчёта используйте токенизатор конкретной модели и актуальный прайс провайдера.

| Класс | Слов | Ток. RU | Ток. EN | Ток. ср. | Ориентир, руб. |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Простые чат-боты или классификация (напр., True/False по заявке) | 200 | 400 | 134 | 266 | 0,08 |
| Сложные корпоративные агенты (напр., ассистент поддержки) | 2 000 | 4 000 | 1 340 | 2 660 | 0,80 |
| Специализированные агенты (напр., юриспруденция, фарма) | 5 000 | 10 000 | 3 350 | 6 650 | 2,00 |
| **Среднее по трём классам** | 2 400 | 4 800 | 1 608 | 3 192 | **0,96** |

#### Класс данных по длине пользовательского текста {: #sizing_data_class_user_text_length }

| Класс | Слов | Ток. RU | Ток. EN | Ток. ср. | Ориентир, руб. |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Короткие (абзац, ответ на вопрос, определение) | 300 | 600 | 201 | 399 | 0,12 |
| Средние (описания, инструкции, устранение неполадок) | 1 500 | 3 000 | 1 005 | 1 995 | 0,60 |
| Длинные (статьи, обзоры, НПА) | 6 000 | 12 000 | 4 020 | 7 980 | 2,39 |
| **Среднее по трём классам** | 2 600 | 5 200 | 1 742 | 3 458 | **1,04** |
```

**Status:** COMPLETED

---

## Step 2: Extend Optimization Table with Risk Column

**Goal:** Add "Риск" column to existing optimization table in "Риски и оптимизация"

**Location:** Line ~843-855 in section `## Риски и оптимизация`

**Original table (restructured lines 847-854):**
```markdown
| Оптимизация | Эффект | Применимость |
| :--- | :--- | :--- |
| DTR-фильтрация | -50% compute | Все LLM |
| SMTL-параллелизм | -70% шагов | Агенты |
| Memex-память | -50% токенов | Длинные задачи |
| KARL-стиль агентного поиска | ≈33% дешевле /≈47% быстрее | Агентный поиск |
| SkillNet-навыки | -30% шагов | Повторяющиеся задачи |
```

**Proposed: Extend with additional optimization + risk context from original lines 633-663**

**From original:** 13-row risk/mitigation table including:
- Раздувание контекста → Жёсткие лимиты, семантическое сжатие
- Галлюцинации → Защитные механизмы, многоуровневая верификация
- Регрессии стека инференса → Закреплять версии, регрессионные проверки
- Мульти-бэкенд → Единая матрица совместимости
- Композитный инцидент → ASVS/WSTG, OWASP Top 10
- etc.

**Question for user:** Should we:
A) Keep existing 5-row table and add admonition with key risk mitigations
B) Replace with expanded table that includes both optimizations AND risks
C) Add a separate H3 "Риски бюджета" with the 13-row table

**Status:** COMPLETED

---

## Step 3: Add Edge-AI and Sovereign Section

**Goal:** Create new H2 `## Автономный инференс` with content about:
- Picoclaw on Raspberry Pi4 (edge-agents)
- Qwen3.5 on M3 Max (consumer hardware for sovereignty)
- CLI vs MCP (protocol choice)

**Content from original:**
- Lines 893-916: Qwen3.5-397B на M3 Max 48ГБ
- Lines 919-968: Picoclaw на Raspberry Pi4 (full section)

**Proposed structure:**
```
## Автономный инференс {: #sizing_autonomous_inference }

### Edge-агенты на минимальном железе {: #sizing_edge_agents_minimal_hardware }
(Picoclaw на Raspberry Pi4)

### Потребительское железо для суверенности {: #sizing_consumer_hardware_sovereignty }
(Qwen3.5 на M3 Max)

### Протоколы для корпоративных систем {: #sizing_protocols_enterprise }
(CLI vs MCP)
```

**Location:** After `## Облачные провайдеры РФ`, before `## TCO и сценарии развёртывания`

**Status:** COMPLETED

---

## Step 4: Add Technical Optimization Admonition

**Goal:** Add admonition with key metrics from Memex, KARL, SMTL, Think@n

**Location:** After optimization table in `### Оптимизация затрат на инференс`

**Content from original (lines 1213-1248):**
- Google Think@n: ~2x reduction in compute
- Oppo AI SMTL: 70.7% fewer inference steps
- Moonshot Attention Residuals: 1.25x computation reduction
- Accenture Memex(RL): -50% peak token consumption
- Databricks KARL: ~33% cheaper, ~47% faster

**Proposed format:** !!! note "Ключевые метрики оптимизации из исследований"

**Status:** READY TO EXECUTE