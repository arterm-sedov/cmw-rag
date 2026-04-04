# Appendix D Enhancement Synthesis

**Date:** March 29, 2026  
**Status:** Ready for Implementation

## Overview

This document synthesizes all research findings into specific enhancement actions for Appendix D (Security Observability).

---

## 1. OWASP Threats Section Enhancements

### Current State (Appendix D)
- References OWASP LLM Top 10 2025 ✓
- References OWASP Agentic Top 10 2026 ✓
- Does NOT enumerate all 10 categories
- No prevalence statistics

### Required Changes

#### 1.1 Add Complete Threat Enumeration Table (After paragraph 32)

New content location: Around line 32-33, after "OWASP Top 10 for LLM Applications (2025)"

**Add table:**
```
## OWASP LLM Top 10 2025 — Полный перечень угроз

| № | Код | Категория | Краткое описание |
|---|-----|-----------|------------------|
| 1 | LLM01:2025 | Prompt Injection | Вредоносный ввод манипулирует поведением LLM |
| 2 | LLM02:2025 | Sensitive Information Disclosure | Модель раскрывает конфиденциальные данные |
| 3 | LLM03:2025 | Supply Chain Vulnerabilities | Скомпрометированные модели, зависимости |
| 4 | LLM04:2025 | Data and Model Poisoning | Манипуляция данными для внедрения бэкдоров |
| 5 | LLM05:2025 | Improper Output Handling | Несанированный вывод LLM передаётся системам |
| 6 | LLM06:2025 | Excessive Agency | LLM предоставлено слишком много функций/прав |
| 7 | LLM07:2025 | System Prompt Leakage | Раскрытие системных инструкций |
| 8 | LLM08:2025 | Vector and Embedding Weaknesses | Уязвимости в механизмах поиска/эмбеддингах |
| 9 | LLM09:2025 | Misinformation | Генерация ложного или вводящего в заблуждение контента |
| 10 | LLM10:2025 | Unbounded Consumption | Чрезмерное потребление ресурсов / DoS |
```

#### 1.2 Add Agentic Top 10 2026 Table (After paragraph 62)

Location: Around line 62-63

**Add:**
```
## OWASP Agentic Top 10 2026 — Полный перечень угроз

| № | Код | Категория | Краткое описание |
|---|-----|-----------|------------------|
| 1 | ASI01:2026 | Agent Goal Hijack | Манипуляция целями агента |
| 2 | ASI02:2026 | Tool Misuse & Exploitation | Опасное использование инструментов |
| 3 | ASI03:2026 | Identity & Privilege Abuse | Компрометация учётных данных агента |
| 4 | ASI04:2026 | Agentic Supply Chain | Скомпрометированные навыки, MCP-серверы |
| 5 | ASI05:2026 | Unexpected Code Execution (RCE) | Выполнение недоверенного кода |
| 6 | ASI06:2026 | Memory & Context Poisoning | Отравление памяти/контекста агента |
| 7 | ASI07:2026 | Insecure Inter-Agent Communication | Атаки на межагентное взаимодействие |
| 8 | ASI08:2026 | Cascading Failures | Каскадные отказы в мультиагентных системах |
| 9 | ASI09:2026 | Human-Agent Trust Exploitation | Социальная инженерия на интерфейсе AI |
| 10 | ASI10:2026 | Rogue Agents | Скомпрометированные агенты |
```

#### 1.3 Add Prevalence Statistics (New subsection)

Location: After OWASP tables, before "Периметр до LLM"

**Add:**
```
### Статистика и реальные инциденты

Промышленные исследования фиксируют значительный уровень уязвимостей в AI-системах:

- **36,82%** AI-навыков содержат уязвимости безопасности (Snyk ToxicSkills, Feb 2026)
- **13,4%** навыков имеют критические проблемы
- **135 000+** экземпляров OpenClaw подвержены атакам
- **36,7%** MCP-серверов уязвимы к SSRF-атакам
- **25%+** навыков в промышленности содержат уязвимости

Источники: Snyk, SecurityScorecard, Antiy CERT (2026)
```

---

## 2. Russian Compliance Section Enhancements

### Current State (Appendix D)
- References 152-FZ correctly ✓
- Mentions AI law is draft (correct as of March 2026) ✓
- No mention of 123-FZ (AI liability law)
- No mention of July 2025 amendments

### Required Changes

#### 2.1 Update paragraph around line 38-39

Current: "Действующая норма по законопроекту об ИИ (на март 2026 — черновик, не вступил в силу)"

**Replace with:**
```
- **152-ФЗ (с поправками 2025 г.):** основной закон о персональных данных; поправки вступили в силу в июле и сентябре 2025 г. — усилены требования к локализации и анонимизации
- **123-ФЗ:** федеральный закон об ответственности за вред, причинённый ИИ (действует с 2024 г.)
- **Стратегия развития ИИ 124/2024:** президентский указ о развитии ИИ до 2030 г.
- **Законопроект о суверенном ИИ:** Минцифры опубликовало проект в марте 2026 г. — требует создания нейросетей в России и обучения на российских данных
```

---

## 3. Observability Section Enhancements

### Current State (Appendix D)
- References OpenTelemetry GenAI semconv ✓
- Mentions Langfuse, Phoenix, LangSmith ✓
- No specific self-hosted recommendations for Russia

### Required Changes

#### 3.1 Enhance observability tools section (around line 96-97)

**Add after existing paragraph:**
```
###_self-hosted_recommendations_for_russia

Для соответствия требованиям 152-ФЗ к локализации обработки ПДн рекомендуютсяself-hosted решения:

| Инструмент | Тип развёртывания | 152-ФЗ соответствие |
|------------|-------------------|---------------------|
| Arize Phoenix | Self-hosted (Docker/K8s) | ✅ Полное |
| Langfuse | Self-hosted (Docker/K8s) | ✅ Полное |
| Helicone | Self-hosted | ✅ Полное |
| SigNoz | Self-hosted | ✅ Полное |
| LangSmith | SaaS (только EU/US) | ❌ Не рекомендуется |

**Рекомендация:** Для российских продакшен-контуров приоритет — self-hosted решения на базе OpenTelemetry с развёртыванием в российских дата-центрах.
```

---

## 4. AI TRiSM Section Enhancements

### Current State (Appendix D)
- References Gartner AI TRiSM ✓
- Mentions OWASP frameworks ✓

### Required Changes

#### 4.1 Add vendor landscape summary (around line 107-108)

**Add:**
```
### Рыночный ландшафт AI TRiSM (2026)

Рынок AI TRiSM оценивается в ~$3,2 млрд (2025), прогноз — $4,83 млрд к 2034 г.

**Ключевые категории поставщиков:**
- **AI Security Platforms:** Palo Alto Networks Prisma AIRS, Microsoft Azure AI Studio
- **AI Governance:** Mindgard, Zenity, PointGuard AI
- **Data Security:** Securiti, Knosti AI
- **Red Teaming:** Microsoft PyRIT, NVIDIA Garak, Giskard

**Тренды 2026:**
- 40% корпоративных приложений будут включать автономных AI-агентов к концу года
- 76% организаций отмечают Shadow AI как значимую проблему
- Только 26% организаций имеют комплексные политики governance AI-безопасности
```

---

## 5. FinOps/Observability Connection

### Current State (Appendix D)
- Mentions token metrics for FinOps ✓

### Enhancement
Add reference to cost observability tools.

---

## Implementation Checklist

- [ ] Add OWASP LLM Top 10 2025 table
- [ ] Add OWASP Agentic Top 10 2026 table
- [ ] Add prevalence statistics
- [ ] Update Russian compliance section with 2025-2026 developments
- [ ] Add self-hosted observability recommendations for Russia
- [ ] Enhance AI TRiSM section with vendor landscape
- [ ] Cross-validate with other report pack documents
- [ ] Final review for coherence

---

## Notes

- All macro figures should be rounded (no 4+ decimal places)
- Final content in Russian
- Sources from deep-research documents should be cited
- Maintain existing document structure and flow
