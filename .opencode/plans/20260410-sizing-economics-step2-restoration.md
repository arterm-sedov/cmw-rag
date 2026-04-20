# Step 2: Restore Detailed Token Calculation Formulas

## Status: IN PROGRESS

## Analysis

### What NEW file has (lines 414-441):
- "Расчёт токенов на слово" table (simplified, no symbols/word column)
- "Расчёт с учётом системного промпта, контекста и рассуждений" table (without "Среднее" row)
- Warning admonition about data relevance

### What NEW file is MISSING:

**1. Calculation Formula Block (OLD lines 520-535):**
```
Слово = ~1,5 токена
Извлечённые данные = 3–10 статей × 500–2500 слов × 1,5 токена/слово
Вход = извлечённые справочные данные + системный промпт + текст заявки
Выход = текст ответа + рассуждения
Всего = вход + выход
Ориентир = Всего × 300 / 1 000 000
```

**2. "Средние длины по корпусу заявок" introductory text (OLD lines 511-520):**
- Empirical model explanation
- Source: "более 12 000 заявок на портале поддержки Comindware"
- RAG breakdown: "80% русский + 20% код/английский"

**3. "Расчёт без учёта системного промпта" table (OLD lines 537-543):**
| Категория | Слов | Ток. RU | Ток. EN | Ток. ср. | Ориентир, руб. |
| Текст заявки | 810 | 1 620 | 543 | 1 077 | 0,32 |
| Текст ответа | 2 641 | 5 282 | 1 769 | 3 513 | 1,05 |
| **Вход+выход** | 3 451 | 6 902 | 2 312 | 4 590 | **1,38** |

**4. "Среднее" row in main table (OLD line 555):**
| **Среднее** | 16 833 | 1 733 | 1 883 | 967 | 21 583 | **6,48** |

**5. Enhanced "Расчёт токенов на слово" table (OLD lines 477-483):**
- Added "Символов на слово" column
- Note about tiktoken validation

**6. More detailed class descriptions in tables:**
- OLD: "Простые чат-боты или классификация (напр., True/False по заявке)"
- NEW: "Простые чат-боты или классификация"

---

## Questions for User

1. **Calculation formula** - Should the formula block be added? It explains how costs are computed from components.

2. **"Расчёт без учёта системного промпта" table** - Should this baseline table be added? Shows raw ticket cost without context overhead.

3. **"Среднее" average row** - Should this be added to the main table? Useful for estimating typical costs.

4. **Enhanced token/word table** - Should "Символов на слово" column be added?

5. **Source attribution** - OLD says "из более чем 12 000 заявок", NEW says "корпус 14 437 заявок". Which is correct?

---

## Placement Options

**Option A:** Add as new subsection after current tables
**Option B:** Integrate into existing "Примерные расчёты расхода токенов" section
**Option C:** Create new "Методология расчёта" H4 subsection