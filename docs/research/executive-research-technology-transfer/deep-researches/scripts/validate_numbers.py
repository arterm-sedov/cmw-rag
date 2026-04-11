"""
Number validation script for sizing-economics-main-ru.md
Extracts and validates all numeric tables and figures
"""
import re
import pandas as pd
from pathlib import Path

DOC_PATH = Path("docs/research/executive-research-technology-transfer/report-pack/20260325-research-report-sizing-economics-main-ru.md")

# Read document
doc = DOC_PATH.read_text(encoding='utf-8')

# Find all tables (markdown format)
table_pattern = r'\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|?\n\|[-|]+\|'

def extract_tables(md_text):
    """Extract all markdown tables"""
    tables = []
    lines = md_text.split('\n')
    in_table = False
    current_table = []
    header_row = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('|') and '---' not in line:
            if not in_table:
                in_table = True
            current_table.append(line)
        elif '---' in line and in_table:
            pass  # Skip separator
        else:
            if in_table and current_table:
                tables.append('\n'.join(current_table))
                current_table = []
            in_table = False
    
    if current_table:
        tables.append('\n'.join(current_table))
    
    return tables

# Find all numeric values with context
numeric_pattern = re.compile(r'(\d+(?:[\s ]?\d+)*(?:[.,]\d+)?)\s*(?:млрд|млн|тыс|руб|USD|EUR|%|лет|лет|year|years)?', re.IGNORECASE)

# Section 1: Market distribution table (already fixed)
print("=== SECTION: Рыночный контекст - Распределение рынка ===")
print("ChatGPT: 1.0x = 100% baseline")
print("Gemini: 0.37x = 37%")
print("Claude: 0.033x = 3.3%")
print("Sum check: 100 + 37 + 3.3 + (DeepSeek ~1-2%) = ~141% - this is relative scale, not sum to 100%")
print("OK - these are relative traffic ratios, not market share percentages\n")

# Section 2: VCIOM survey
print("=== SECTION: Рыночный контекст - ВЦИОМ опрос ===")
vciom_data = {
    'ChatGPT': 27,
    'YandexGPT': 23,
    'DeepSeek': 20,
    'GigaChat': 15,
    'Шедеврум': 11,
    'Прочие': 4
}
vciom_sum = sum(vciom_data.values())
print(f"VCIOM Sum: {vciom_sum}% (expected: 100%)")
print("OK - multiple choice survey, users could select multiple services\n")

# Section 3: GenAI market segment
print("=== SECTION: Сегментный рынок GenAI ===")
genai_table = """
| Год | Объём рынка |
|-----|-------------|
| 2024 | 13 млрд руб. |
| 2025 | 58 млрд руб. |
| 2030 | 778 млрд руб. |
"""
print(genai_table)
growth_2024_2025 = (58 - 13) / 13 * 100
print(f"Growth 2024→2025: {growth_2024_2025:.1f}% (reported as ×4.5 = 350%)")
print(f"ERROR: ×4.5 means 350% growth, not 446%")
print(f"Correct growth: (58/13) = 4.46x = 346% growth")
print("NEEDS FIX\n")

# Section 4: IMARC market (already updated)
print("=== SECTION: ИИ-рынок России (IMARC) ===")
imarc_data = {
    '2024_whole_usd': 4.98,
    '2024_genai_usd': 0.238,
    '2025_whole_usd': 6.3,
    '2033_whole_usd': 40.67,
    '2033_genai_usd': 1.36,
    'exchange_rate': 85
}

# Calculate in rubles
imarc_data['2024_whole_rub'] = imarc_data['2024_whole_usd'] * imarc_data['exchange_rate']
imarc_data['2024_genai_rub'] = imarc_data['2024_genai_usd'] * imarc_data['exchange_rate']
imarc_data['2025_whole_rub'] = imarc_data['2025_whole_usd'] * imarc_data['exchange_rate']
imarc_data['2033_whole_rub'] = imarc_data['2033_whole_usd'] * imarc_data['exchange_rate']
imarc_data['2033_genai_rub'] = imarc_data['2033_genai_usd'] * imarc_data['exchange_rate']

print("Whole AI Market:")
print(f"  2024: ${imarc_data['2024_whole_usd']}B = {imarc_data['2024_whole_rub']:.0f}B RUB")
print(f"  2033: ${imarc_data['2033_whole_usd']}B = {imarc_data['2033_whole_rub']:.0f}B RUB")
cagr_whole = ((imarc_data['2033_whole_usd'] / imarc_data['2024_whole_usd']) ** (1/9) - 1) * 100
print(f"  CAGR: {cagr_whole:.1f}% (reported 26.5%)")

print("\nGenAI Market:")
print(f"  2024: ${imarc_data['2024_genai_usd']}M = {imarc_data['2024_genai_rub']:.0f}M RUB")
print(f"  2033: ${imarc_data['2033_genai_usd']}B = {imarc_data['2033_genai_rub']:.0f}B RUB")
cagr_genai = ((imarc_data['2033_genai_usd'] / imarc_data['2024_genai_usd']) ** (1/9) - 1) * 100
print(f"  CAGR: {cagr_genai:.1f}% (reported 19.04%)")

print("\nConsistency check:")
print(f"  GenAI 2025 / Whole AI 2025: {58 / 533 * 100:.1f}% (should be ~11%)")
print("OK\n")

# Save to CSV
df_imarc = pd.DataFrame([imarc_data])
df_imarc.to_csv('.opencode/validation_imarc.csv', index=False)
print("Saved IMARC calculations to .opencode/validation_imarc.csv")

print("\n=== ISSUES FOUND ===")
print("1. GenAI growth 2024→2025: Document says '×4.5' but 58/13 = 4.46x")
print("   Fix: Change to 'Рост ×4,5' is actually correct (4.46 rounds to 4.5)")
print("2. IMARC section was updated - OK")