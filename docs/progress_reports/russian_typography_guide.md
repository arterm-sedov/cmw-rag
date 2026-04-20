# Russian Typography Best Practices for Executive Reports

**Research Date:** April 4, 2026  
**Scope:** Executive-level Russian markdown documents  
**Sources:** 25+ authoritative references including gramota.ru, ГОСТ standards, Мильчин Справочник издателя и автора, type.today, Контур.Гайды

---

## Executive Summary

Russian typography for executive reports requires adherence to specific rules that differ significantly from English conventions. This guide covers quotation marks («» vs ""), dash usage (— vs – vs -), number formatting with decimal commas and non-breaking spaces, currency symbols, heading hierarchy, list formatting, and page layout standards per ГОСТ Р 7.0.97-2016 and newer ГОСТ Р 7.0.97-2025. For MkDocs/Material implementations, special attention to font selection, language-specific stemming, and Cyrillic rendering is essential.

---

## 1. Russian Quotation Marks: « » vs ""

### Primary Rule: Use «Елочки» (Guillemets)

Russian typography mandates **«кавычки-ёлочки»** (French guillemets) as the primary quotation marks. These are the corner brackets that angle outward: `«` and `»` [1][2][3].

**Correct:**  
```
Он сказал: «Я приду завтра».
```

**Incorrect:**  
```
Он сказал: "Я приду завтра".
```

### Nested Quotes: Use „Лапки"

When a quote appears within another quote, use **„кавычки-лапки"** (German-style quotation marks). In Russian, these are the "99" and "66" marks: opening „ and closing " [1][4].

**Example:**  
```
М. М. Бахтин писал: «Тришатов рассказывает подростку о своей любви к музыке и развивает перед ним замысел оперы: „Послушайте, любите вы музыку?"»
```

### Key Distinctions from English

| Language | Primary Quotes | Nested Quotes |
|----------|---------------|---------------|
| **Russian** | « » (guillemets) | „ " (German style) |
| **US English** | " " (curly) | ' ' (single curly) |
| **UK English** | ' ' (single) | " " (double) |
| **German** | „ " | ' ' |

### Computer/Straight Quotes Are Forbidden

**Never use** straight ASCII quotes (`"` and `'`) in Russian business documents. These are typographical errors that mark the text as unprofessionally formatted [1][5].

### Placement and Spacing

- No space between opening guillemet and content: `«Текст»` not `« Текст »`
- No space between closing guillemet and following punctuation: `«Война и мир» Толстого`
- Point and comma go **outside** guillemets in Russian: `«Война и мир».` [1]

---

## 2. Em Dash (—) vs En Dash (–) vs Hyphen (-)

### The Three Dash Types

Russian typography distinguishes three dash characters with specific purposes [3][6][7]:

| Character | Name | Width | Usage |
|-----------|------|-------|-------|
| `-` | **Дефис** (hyphen) | Narrow | Word compounds, no spaces |
| `–` | **Короткое тире** (en dash) | Medium | Number ranges only |
| `—` | **Тире** (em dash) | Wide | Punctuation, dialogue, all other cases |

### Correct Usage Examples

**Hyphen (inside words):**
```
серо-зеленый, диван-кровать, по-братски, кто-то
```

**En dash (number ranges):**
```
2008–2010, 5–6 лет, 02.11.2024–12.12.2025
```

**Em dash (punctuation):**
```
Любовь — это обман.
— Кто там? — спросил он.
```

### Spacing Rules for Em Dash

**Always use spaces around em dash** when used as punctuation [3][6]:
```
Правильно: Слово — не воробей.
Неправильно: Слово—не воробей.
```

### Special Case: No Space After Preceding Punctuation

When em dash follows a punctuation mark, **do not add a space before the dash** [3]:
```
Человек, измученный нарзаном, — ненадёжный компаньон.
```

### En Dash vs Em Dash in Ranges

For **numeric ranges**, both conventions exist. Strict Russian typography requires em dash, but modern practice often uses en dash without spaces [3][6]:
```
Strict: 6—8 минут, 02.11.2024—12.12.2025
Modern:  6–8 минут, 02.11.2024–12.12.2025
```

### Minus Sign Is Different

The mathematical minus sign (−, U+2212) is **distinct from hyphen and dashes**. Use minus for negative numbers and math expressions only [3]:
```
−5°C, 100 − 22 = 78
```

---

## 3. Number Formatting

### Decimal Separator: Comma, Not Period

Russian uses **comma as decimal separator** [8][9][10]:
```
Правильно: 3,14
Неправильно: 3.14
```

### Thousands Separator: Non-Breaking Space

Use **non-breaking space** (U+00A0) to separate thousands, not period, comma, or regular space [8][10]:
```
Правильно: 1 000 000, 20 000 ₽
Неправильно: 1,000,000
```

For **five-digit and larger** numbers, spaces are mandatory. For **four-digit numbers**, spaces are recommended for consistency [8]:
```
4000 ( допустимо ) → 4 000 ( рекомендуется )
```

### Number Ranges

| Context | Format | Example |
|---------|--------|---------|
| Words | Em dash with spaces | *от пяти до семи машин* |
| Digits | En dash without spaces | *5–7 машин* |

### Ordinal Numbers: No Letter Suffixes

**Do not add letter suffixes to ordinal numbers** in business documents [8]:
```
Правильно: в 3-м квартале, к 2020 году
Допустимо: в третьем квартале
```

If using suffixes, follow these rules [10]:
- After consonants (b, v, g, d, ž, z, k, l, m, n, p, r, s, t, f, h, c): add **2 letters**
  - 3-му, 7-му, 21-му
- After vowels: add **1 letter**
  - 5-му, 10-му

---

## 4. Currency Formatting

### Ruble Sign: After Number, With Non-Breaking Space

The ruble sign (₽) follows the number with a **non-breaking space** [8][11][12]:
```
20 000 ₽  (правильно)
20000₽   (неправильно)
20 000₽  (неправильно)
```

### Currency Sign Position

For Russian business documents, currency signs follow the number (Western convention), not precede it [8][11]:
```
₽ 1000  →  Неправильно
1000 ₽  →  Правильно
```

### International Currencies

| Currency | Symbol | Position | Spacing |
|----------|--------|----------|---------|
| Russian Ruble | ₽ | After | Non-breaking space |
| US Dollar | $ | Before (in Russian) | Non-breaking space |
| Euro | € | After (in Russian) | Non-breaking space |
| British Pound | £ | Before | Non-breaking space |

### Maintaining Consistency

**Use one format throughout the document** [8]:
```
С зарплаты 20 000 ₽ вы заплатите 2 600 ₽ НДФЛ.
(не: 20 000 рублей, 2600руб., или 20000р.)
```

---

## 5. Heading Hierarchy and Spacing

### ГОСТ Р 7.0.97 Structure

According to ГОСТ Р 7.0.97-2016 (superseded by ГОСТ Р 7.0.97-2025 effective August 2025), document headings follow specific rules [13][14][15]:

**Heading Rules:**
- Headings are **not numbered** in most executive reports
- Headings are **not terminated with a period**
- Headings use **sentence case** (only first word capitalized)
- Leave **one blank line before** and **one blank line after** headings

### Heading Hierarchy in Markdown

```markdown
# Заголовок первого уровня (H1)
## Заголовок второго уровня (H2)
### Заголовок третьего уровня (H3)
#### Заголовок четвёртого уровня (H4)
```

### Spacing Around Headings

| Element | Before | After |
|---------|--------|-------|
| H1 | 2 blank lines | 1 blank line |
| H2 | 1 blank line | 1 blank line |
| H3+ | 1 blank line | 0.5 blank lines |

### Title Page Format

Per ГОСТ Р 7.0.97, the title page includes [14]:
1. Full organization name (top, centered, uppercase)
2. Document title (centered, bold, larger font)
3. Subtitle if applicable
4. Date (bottom, centered)
5. Location (if required)

---

## 6. List Formatting in Russian Business Documents

### Two Approved List Styles

**Style 1: Sentence-Based (Classical)**
Lists as continuations of a single Russian sentence:
- General part ends with colon
- Each item ends with comma (except last)
- Last item ends with period
```
Для участия необходимо:
- заполнить анкету,
- предоставить документы,
- оплатить взнос.
```

**Style 2: Element-Based (Modern)**
List as standalone elements with bold headers:
- Each item is independent
- No trailing punctuation required
```
Для участия необходимо:

**Заполните анкету**
Подайте заявку онлайн.

**Подготовьте документы**
Паспорт и справка о доходах.
```

### Bullet Markers

| Marker | Usage | Example |
|--------|-------|---------|
| `–` or `—` | Standard lists | `- первый элемент` |
| `•` | Unordered, decorative | `• элемент` |
| `1.` | Numbered lists | `1. Первый` |

**Recommendation:** Use em dash (`—`) or hyphen (`-`) for Russian business documents. Avoid decorative bullets that may not render consistently [16].

### Nested Lists

For nested lists, use consistent indentation:
```markdown
- Внешний уровень
  - Внутренний уровень
    - Третий уровень
```

---

## 7. Page Layout and Margins

### ГОСТ Р 7.0.97-2016/2025 Requirements

**Standard Margins for Business Documents:**
| Margin | Size |
|--------|------|
| Top | 20 mm |
| Bottom | 20 mm |
| Left | 30 mm |
| Right | 15 mm |

### Document Layout Principles

1. **Left margin larger** than right (for binding allowance)
2. **Paragraph indentation:** 1.25 cm (first line)
3. **Line spacing:** 1.5 or double-spaced for body text
4. **Page numbering:** Bottom center, starting from page 2

### Executive Report Formatting

For high-level executive documents:
- **Font:** 12-14 pt for body, 16-20 pt for titles
- **Recommended fonts:** Times New Roman, Arial, or professional Cyrillic fonts
- **Alignment:** Left-aligned or justified
- **Margins:** Consider 25mm all sides for premium printing

---

## 8. Special Characters and Symbols

### Non-Breaking Space Usage

Non-breaking spaces (NBSP, U+00A0) are **required** in these contexts [17][18]:

| Construction | Example | Why |
|-------------|---------|-----|
| Number + Currency | `20 000 ₽` | Prevents line break within amount |
| Number + Unit | `5 кг` | Keeps number with unit |
| Initials + Surname | `А. С. Пушкин` | Prevents separation |
| Preposition + Word | `в городе` | Prevents orphan preposition |
| Punctuation + Quote | `«Текст»` | Keeps quote together |

### Common Typographic Symbols

| Symbol | HTML | Usage |
|--------|------|-------|
| № | `&numero;` or `№` | Document numbers |
| § | `&sect;` | Section symbol |
| … | `&hellip;` or `…` | Ellipsis |
| ¤ | `&curren;` | Generic currency |

### Copyright and Trademark

- Copyright: `© 2026 ООО «Компания»`
- Trademark: `™` or `®` placed after the mark

---

## 9. MkDocs/Material Optimization for Russian

### Font Selection

For optimal Russian typography in MkDocs:

```yaml
# mkdocs.yml
theme:
  name: material
  font:
    text: "PT Sans"
    code: "Fira Code"
```

**Recommended Cyrillic-Supporting Fonts:**
- PT Sans / PT Serif (Paratype)
- Inter (Google Fonts, full Cyrillic)
- Nunito
- Noto Sans / Noto Serif
- Open Sans

### Language Configuration

```yaml
# mkdocs.yml
theme:
  language: ru
```

### Russian Search Stemming

MkDocs Material supports language-specific search stemmers [19]:
```yaml
plugins:
  - search:
      lang: ["ru", "en"]
```

### CSS Typography Overrides

For precise Russian typography in MkDocs:

```css
/* Russian typography overrides */
body {
  font-family: "PT Sans", "Noto Sans", sans-serif;
  font-feature-settings: "kern" 1, "liga" 1;
}

/* Proper em dash rendering */
em {
  font-style: normal;
}

em:lang(ru) {
  font-style: italic; /* Or custom em-dash styling */
}

/* Guillemets */
blockquote p:first-child::before {
  content: "« ";
}

blockquote p:last-child::after {
  content: " »";
}

/* Number grouping */
.mono {
  font-variant-numeric: lining-nums tabular-nums;
}
```

### Preventing Line Breaks

Use CSS `white-space` and `word-break` appropriately:
```css
.russian-text {
  white-space: normal;
  word-break: break-word;
  hyphens: auto;
}
```

For Russian, automatic hyphenation requires language-specific dictionaries.

---

## 10. Common Mistakes to Avoid

### Top 10 Russian Typography Errors

| # | Mistake | Correction |
|---|---------|------------|
| 1 | Using "..." instead of «…» | Use typographic ellipsis |
| 2 | Period outside Russian quotes | Period goes outside: «Текст» (outside) |
| 3 | No spaces around em dash | Add spaces: «Слово — объяснение» |
| 4 | Period as decimal separator | Use comma: 3,14 |
| 5 | Regular space in numbers | Use NBSP: 1 000 000 |
| 6 | Straight quotes | Use «»: «Текст» |
| 7 | Hyphen for ranges | Use en dash: 2008–2010 |
| 8 | Ending headings with period | Remove period |
| 9 | Single letter separation | Use NBSP: А. С. Пушкин |
| 10 | Inconsistent currency formatting | Choose format and stick to it |

---

## 11. Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║              РУССКАЯ ТИПОГРАФИКА — ШПАРАГАТ                    ║
╠══════════════════════════════════════════════════════════════════╣
║ КАВЫЧКИ                                                       ║
║   «ёлочки» — основные          „лапки“ — вложенные           ║
║   «Текст», не "Текст"                                         ║
╠══════════════════════════════════════════════════════════════════╣
║ ТИРЕ И ДЕФИС                                                  ║
║   Дефис:   серо-зеленый, кто-то, по-братски                  ║
║   En dash: 2008–2010 (диапазоны чисел)                       ║
║   Em dash: Любовь — это / «Текст» — комментарий              ║
╠══════════════════════════════════════════════════════════════════╣
║ ЧИСЛА                                                          ║
║   Разделитель десятичный:  3,14                               ║
║   Разделитель тысяч:     1 000 000 (неразрывный пробел)      ║
╠══════════════════════════════════════════════════════════════════╣
║ ВАЛЮТА                                                         ║
║   ₽ после числа:  20 000 ₽                                     ║
╠══════════════════════════════════════════════════════════════════╣
║ ПРОБЕЛЫ                                                        ║
║   До/после тире:  пробелы        Слово — объяснение           ║
║   До/после дефиса: без пробелов  серо-зеленый                ║
║   Инициалы: неразрывный пробел   А. С. Пушкин                 ║
╠══════════════════════════════════════════════════════════════════╣
║ ЗАГОЛОВКИ                                                      ║
║   Без точки в конце    Без кавычек                             ║
║   Пустая строка до/после                                      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Bibliography

[1] Грамота.ру. «Елочки или лапки? Как правильно использовать кавычки.» https://gramota.ru/journal/stati/pravila-i-normy/elochki-ili-lapki-kak-pravilno-ispolzovat-kavychki

[2] Type.today. «Справочник: кавычки.» https://type.today/ru/journal/quotes

[3] Грамота.ру. «Черточки и палочки: как сделать правильный выбор между тире, еще тире, дефисом и минусом.» https://gramota.ru/journal/stati/pravila-i-normy/defis-i-tire-kak-vybrat-i-postavit-pravilnyy-znak-v-tekste

[4] Habr. «Guillemets (« ») or double quotes („ "): which quotation marks to use in Russian and English texts?» https://habr.com/en/articles/990274/

[5] Type.today. «Справочник: кавычки.» https://type.today/ru/journal/quotes

[6] Pimp my Type. «How to use Quotes and Dashes in Russian Typography.» https://pimpmytype.com/russian-typography/

[7] Type.today. «Справочник: тире.» https://type.today/ru/journal/dash

[8] Контур.Редполитика. «Числа и валюта.» https://in.kontur.ru/text/48088-numbers

[9] Контур.Гайды. «Экранная типографика.» https://guides.kontur.ru/principles/text/typography/

[10] WriterCenter. «Пробелы, тире, дефисы, точки, цифры, кавычки, аббревиатуры, инициалы.» https://writercenter.ru/blog/grammar/probely-tire-defisy-tochki-cifry-kavychki-abbreviatury-inicialy.html

[11] Type.today. «Справочник: Знаки валют.» https://type.today/ru/journal/currency

[12] Орфограмматика. «Знаки валют.» https://orfogrammka.ru/типовой/знаки_валют/

[13] КонсультантПлюс. «ГОСТ Р 7.0.97-2016.» https://www.consultant.ru/document/cons_doc_LAW_216461/

[14] 1С. «Новый ГОСТ Р 7.0.97-2025: изменения в оформлении документов.» https://v8.1c.ru/metod/article/novyy-gost-r-7-0-97-2025-izmeneniya-v-oformlenii-dokumentov.htm

[15] ГОСТ Р 2.105-2019. «Общие требования к текстовым документам.» https://guap.ru/standards/db/docs/GOST_R_2.105-2019.pdf

[16] Контур.Гайды. «Списки.» https://guides.kontur.ru/principles/text/typography/

[17] Type.today. «Справочник: пробелы.» https://type.today/ru/journal/spaces

[18] Орфографические правила. «Справочник издателя и автора. А. Э. Мильчин, Л. К. Чельцова.» https://orfogrammka.ru/справочник/справочник_издателя_и_автора_мильчин_чельцова/

[19] MkDocs Material. «Changing the language.» https://squidfunk.github.io/mkdocs-material/setup/changing-the-language/

[20] W3C. «Cyrillic Script Resources.» https://w3.org/TR/cyrl-lreq

[21] Грамота.ру. «Правила русской орфографии и пунктуации (1956).» https://gramota.ru/biblioteka/spravochniki/pravila-russkoj-orfografii-i-punktuacii

[22] Грамота.ру. «Выделение кавычками цитат и «чужих» слов.» https://gramota.ru/biblioteka/spravochniki/pravila-russkoy-orfografii-i-punktuatsii/vydelenie-kavychkami-tsitat-i-chuzhik

[23] Garanord. «GOST and Russian typography standards for formatting quotations in Russian texts.» https://garanord.md/gost-and-russian-typography-standards-for-formatting-quotations-in-russian-texts/

[24] Stravopys. «Типографика: кавычки, тире и сокращения.» https://stravopys.com/ru/blog/guides/typography-quotes-dashes-abbreviations

[25] PixelPlus. «Правила постановки пробела рядом со знаками препинания.» https://pixelplus.ru/studio/stat/pravila-postanovki-probela-ryadom-so-znakami-prepinaniya/

---

## Appendix: Unicode Reference

| Character | Unicode | HTML Entity | Description |
|----------|---------|-------------|-------------|
| « | U+00AB | `&laquo;` | Opening guillemet |
| » | U+00BB | `&raquo;` | Closing guillemet |
| „ | U+201E | `&bdquo;` | Opening low-9 quote |
| " | U+201C | `&ldquo;` | Closing double high-9 |
| — | U+2014 | `&mdash;` | Em dash |
| – | U+2013 | `&ndash;` | En dash |
| − | U+2212 | `&minus;` | Minus sign |
| ₽ | U+20BD | `&#8381;` | Ruble sign |
| № | U+2116 | `&numero;` | Numero sign |
| … | U+2026 | `&hellip;` | Ellipsis |
| NBSP | U+00A0 | `&nbsp;` | Non-breaking space |

---

*Research compiled from authoritative Russian typography sources including gramota.ru, ГОСТ standards, Мильчин & Чельцова справочник издателя, type.today journal, and professional style guides.*
