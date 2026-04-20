# Steps 4-6 Analysis: Missing Content

## Summary

| # | Section | OLD Location | Status in NEW |
|---|---------|--------------|---------------|
| 4 | Анализ чувствительности | Lines 618-629 | **MISSING** |
| 5 | Ориентиры для глубокого аппаратного сайзинга | Lines 692-697 | **BROKEN LINK** - anchor exists but content missing |
| 6 | Шаг 3 — сопоставление (TCO explanation) | Line 1067 | **MISSING** |

---

## Step 4: "Анализ чувствительности" Table

### OLD Content (Lines 618-629):
```markdown
### Анализ чувствительности {: #sizing_sensitivity_analysis }

- Рост длины контекста на 2x увеличивает OpEx в облаке на 1.8x, но почти не влияет на On-Prem (до предела VRAM).
- Квантование (Q4) снижает требования к VRAM в 4 раза при потере точности < 3%.

| Параметр | Консервативный (Small) | Базовый (Medium) | Агрессивный (Enterprise) |
| **Нагрузка (DAU)** | 10–50 пользователей | 50–500 пользователей | 500+ пользователей |
| **Запросов/день** | ~200 | ~2 500 | ~10 000+ |
| **Средний контекст** | 4K токенов | 16K токенов | 32K–128K токенов |
| **Норматив задержки** | < 5 с | < 2 с | < 1 с (реальное время) |
| **Рекомендуемое железо** | 1× RTX 4090 / аналог | 2×–4× RTX 4090 или A100 | GPU-кластер |
```

### Value Analysis:
- **High value:** Practical guidance for sizing decisions
- **Links:** References on-prem GPU profile - useful for decision-making
- **Clear:** Three tiers (Small/Medium/Enterprise) match business scenarios

---

## Step 5: "Ориентиры для глубокого аппаратного сайзинга" (BROKEN LINK)

### OLD Content (Lines 692-697):
```markdown
#### Ориентиры для глубокого аппаратного сайзинга (официальные бенчмарки и документация) {: #sizing_hardware_deep_research_pointers }

Для глубокого аппаратного сайзинга опирайтесь на **официальные** и воспроизводимые источники: отраслевые бенчмарки **MLCommons**, документацию и release notes **vLLM**, а также VRAM-калькуляторы как вспомогательный инструмент для грубой прикидки. Эти источники помогают не подменять инженерную проверку маркетинговыми цифрами и отделять оценку **железа**, **KV-кэша**, **батча** и **перерасхода памяти фреймворка**.
Оценки CapEx и аренды сверяйте с [оценкой пропускной способности] и [локальным инференсом LLM].
Юридические нюансы по поставке GPU — см. Приложение B.
```

### NEW File Status:
- Line 582 has: `Контекст по официальным бенчмаркам и документации — _«[Ориентиры для глубокого аппаратного сайзинга](#sizing_hardware_deep_research_pointers)»_.`
- But the anchor `#sizing_hardware_deep_research_pointers` **does not exist** in the NEW file!
- This is a **broken internal link**

### Value Analysis:
- **Medium value:** Points users to authoritative sources
- **BROKEN:** Must be fixed - either restore content or remove link

---

## Step 6: "Шаг 3 — сопоставление" (TCO Explanation)

### OLD Content (Line 1067):
```markdown
**Шаг 3 — сопоставление:** приведённый CapEx **~10–20 млн** относится к **закупке**; ряд **~179–463 млн** — к **трёхлетней** аренде при **полной** занятости по часам выше. Отсюда **не** следует один коэффициент «экономии 70%» без профиля часов: при снижении фактических часов аренды облако выигрывает; при **высокой** устойчивой утилизации и учёте только «цены железа» выигрывает закупка. Полный on-prem TCO заказчик собирает из CapEx узла, электроэнергии, обслуживания и риска устаревания...
```

### NEW File Status:
- Has "Шаг 1" and "Шаг 2" (lines 883, 892)
- **MISSING "Шаг 3"** - jumps from Шаг 2 to "Порог утилизации"

### Value Analysis:
- **High value:** Critical warning about not comparing raw numbers
- **Explains:** Why you can't just divide 179/10 and say "70% savings"
- **Provides context:** What full TCO actually includes

---

## Recommended Actions

1. **Step 4 (Sensitivity Analysis):** Insert table after "Минимальные системные требования" or before "TCO и сценарии развёртывания"

2. **Step 5 (Deep Hardware Pointers):** Either:
   - A: Restore the full explanation with links
   - B: Remove the broken cross-reference link

3. **Step 6 (TCO Explanation):** Insert "Шаг 3 — сопоставление" after "Шаг 2" in TCO section