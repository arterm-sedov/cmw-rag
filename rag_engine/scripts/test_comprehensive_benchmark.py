"""
COMPREHENSIVE BENCHMARK: 2-Stage SLM+REGEX vs 3-Stage BERT Pipeline
Full 100-sample test, EN + RU, Open-source models only

Models tested:
- qwen/qwen3-8b (Best overall, 119 langs)
- qwen/qwen3-4b:free (Free tier, 119 langs)
- qwen/qwen3-1.7b:free (Smallest Qwen3, very fast)
- openai/gpt-oss-20b (Open-source GPT, comparison baseline)
- google/gemma-3-4b-it (Multilingual, good balance)
"""

import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATASET_PATH = r"D:\Repo\cmw-rag\rag_engine\data\synthetic_cmw_support_tickets_v1_with_pii.json"

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# =============================================================================
# COMPREHENSIVE PROMPT (with Extraction Rules + 5 Examples)
# =============================================================================

PROMPT_COMPREHENSIVE = """You are a PII extractor for IT SUPPORT TICKETS.

Your job is to extract ALL sensitive information from technical support requests.

## Extraction Rules:
1. **Be Aggressive:** Extract anything that looks like personal or corporate PII. High recall is critical.
2. **Context Matters:** Only extract technical strings (IPs, Logins, Keys) if they represent specific instances, not placeholders.
3. **Handle Technical Noise:**
   - DO NOT extract software names as companies (e.g., Kafka, Elasticsearch, Nginx, Docker, Kubernetes are NOT companies).
   - DO NOT extract version numbers (e.g., 5.0.13334) as ZIP codes or INNs.
   - DO NOT extract local/loopback addresses (127.0.0.1, localhost) unless specifically instructed.
4. **Full Entities:** Extract full names (John Smith) and full organizational titles.
5. **No Fictional Data:** If the input contains "example.com" or "test@test.com", skip them.

## Entity Types (25 classes)

Extract these EXACT types:
1. NAME - Person names (Иван Иванов, John Smith, Петров В.А., Sarah Wilson)
2. COMPANY - Organizations (ООО Ромашка, АльфаТеч, Acme IT, Microsoft)
3. EMAIL - Email addresses (user@example.com, ivanov@company.ru)
4. PHONE - Phone numbers in ANY format
5. INN - Russian Tax ID (10-12 digits)
6. IP_ADDRESS - IP addresses (IPv4, IPv6)
7. URL - Web links (https://support.example.com/ticket/123)
8. LOGIN - Usernames, accounts (admin, systemAccount, service_account)
9. PASSWORD - Passwords visible in text
10. API_KEY - API keys, tokens, secrets
11. ADDRESS - Full addresses
12. CITY - City names
13. BANK_CARD - Credit card numbers
14. BIC - Bank BIC/SWIFT codes
15. CAR_NUMBER - Vehicle plate numbers
16. PASSPORT - Passport numbers
17. SNILS - Russian SNILS
18. OGRN - Russian OGRN number
19. DATE - Dates
20. VERSION - Software versions (5.0.13334.0, v2.1.0)
21. US_SSN - US Social Security Number (XXX-XX-XXXX)
22. US_ZIP - US ZIP Code (12345, 12345-6789)
23. US_DRIVER_LICENSE - US Driver License
24. US_EIN - US Employer ID (XX-XXXXXXX)
25. US_NPI - US National Provider Identifier

## Use REGEX to find patterns

You're good at coding - use regex to find these patterns:

```
# Phone: Russian
\\+7\\s*\\(?\\d{3}\\)?\\s*\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{2}
8\\s*\\(?\\d{3}\\)?\\s*\\d{3}[-\\s]?\\d{2}[-\\s]?\\d{2}

# Phone: US/International
\\+?1?[-.\\s]?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}

# Email
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}

# IP Address
\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b

# Russian INN (10-12 digits, context: инн, inn)
\\b\\d{10}\\b|\\b\\d{12}\\b

# Russian Passport (series + number)
\\b\\d{4}\\s*\\d{6}\\b

# US SSN
\\b\\d{3}-\\d{2}-\\d{4}\\b

# Bank Card (16 digits)
\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b

# URL
https?://[^\\s/$.?#].[^\\s]*

# Version numbers
\\b\\d+\\.\\d+(\\.\\d+)?\\b
```

## Examples (RU + EN, IT Support Context)

Example 1 (RU):
Input: "Добрый день! Пользователь Иван Иванов из компании ООО Ромашка сообщил о проблеме. Email: ivanov@romashka.ru, телефон: +7 495 123-45-67 доб. 100. Сервер 192.168.1.10 недоступен. Версия платформы: 5.0.13334.0. ИНН организации: 7714012345."
Output: [
  {"text": "Иван Иванов", "type": "NAME"},
  {"text": "ООО Ромашка", "type": "COMPANY"},
  {"text": "ivanov@romashka.ru", "type": "EMAIL"},
  {"text": "+7 495 123-45-67 доб. 100", "type": "PHONE"},
  {"text": "192.168.1.10", "type": "IP_ADDRESS"},
  {"text": "5.0.13334.0", "type": "VERSION"},
  {"text": "7714012345", "type": "INN"}
]

Example 2 (EN):
Input: "Contact John Smith at john.smith@company.com. Server 10.0.0.50 has issues. US SSN: 123-45-6789. ZIP: 90210. See https://support.example.com/ticket/12345. Version 2.1.0"
Output: [
  {"text": "John Smith", "type": "NAME"},
  {"text": "john.smith@company.com", "type": "EMAIL"},
  {"text": "10.0.0.50", "type": "IP_ADDRESS"},
  {"text": "123-45-6789", "type": "US_SSN"},
  {"text": "90210", "type": "US_ZIP"},
  {"text": "https://support.example.com/ticket/12345", "type": "URL"},
  {"text": "2.1.0", "type": "VERSION"}
]

Example 3 (Mixed):
Input: "Account systemAccount is locked. INN: 1234567890. Bank: 30101810400000000225. Card: 4111 1111 1111 1111."
Output: [
  {"text": "systemAccount", "type": "LOGIN"},
  {"text": "1234567890", "type": "INN"},
  {"text": "30101810400000000225", "type": "BANK_ACCOUNT"},
  {"text": "4111 1111 1111 1111", "type": "BANK_CARD"}
]

Example 4 (RU Company docs):
Input: "ОГРН: 1167746071421, КПП: 771401001, БИК: 044525201. Паспорт: 1234 567890."
Output: [
  {"text": "1167746071421", "type": "OGRN"},
  {"text": "771401001", "type": "KPP"},
  {"text": "044525201", "type": "BIC"},
  {"text": "1234 567890", "type": "PASSPORT"}
]

Example 5 (US Medical):
Input: "Provider NPI: 1234567890. EIN: 12-3456789. Driver license: A1234567. Address: 123 Main St, New York, NY 10001."
Output: [
  {"text": "1234567890", "type": "US_NPI"},
  {"text": "12-3456789", "type": "US_EIN"},
  {"text": "A1234567", "type": "US_DRIVER_LICENSE"},
  {"text": "123 Main St, New York, NY 10001", "type": "ADDRESS"}
]

## Output Format
Return ONLY a valid JSON array. No markdown, no explanation.
Extract EVERY entity you can find. Be aggressive! Use regex to find patterns.

Input: """


# =============================================================================
# JSON Schema Structured Output
# =============================================================================

def extract_with_json_schema(text: str, model: str) -> list[dict]:
    """Extract PII using JSON Schema structured output."""
    try:
        json_schema = {
            "name": "pii_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "type": {"type": "string"}
                            },
                            "required": ["text", "type"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["entities"],
                "additionalProperties": False
            }
        }

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_COMPREHENSIVE},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )
        
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        
        if isinstance(data, dict) and "entities" in data:
            return data["entities"]
        return []
    except Exception as e:
        return []


# =============================================================================
# REGEX DETECTOR
# =============================================================================

class RobustRegexDetector:
    """Context-aware regex detection."""
    
    PATTERNS = [
        (r'(?:пароль|password|passwd|pwd)[\s:=_]+(\S+)', "PASSWORD", None, 100),
        (r'(?:логин|login|username)[\s:=_]+(\S+)', "LOGIN", None, 100),
        (r'(?:api[_-]?key|apikey)[\s:=_]+(\S+)', "API_KEY", None, 100),
        (r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}\b', "SNILS", ['снилс', 'snils'], 90),
        (r'\b\d{10}\b', "INN", ['инн', 'inn', 'налог'], 85),
        (r'\b\d{12}\b', "INN", ['инн', 'inn', 'налог'], 85),
        (r'\b\d{13}\b', "OGRN", ['огрн', 'ogrn'], 85),
        (r'\b\d{15}\b', "OGRN", ['огрн', 'ogrn'], 85),
        (r'\b\d{9}\b', "KPP", ['кпп', 'kpp'], 80),
        (r'\b\d{4}\s*\d{6}\b', "PASSPORT", None, 85),
        (r'\b0[4-5]\d{7}\b', "BIC", ['бик', 'bic'], 85),
        (r'\b40[5-8]\d{17}\b', "BANK_ACCOUNT", None, 90),
        (r'\b301\d{17}\b', "BANK_ACCOUNT", None, 90),
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "BANK_CARD", None, 80),
        (r'\b\d{3}-\d{2}-\d{4}\b', "US_SSN", None, 95),
        (r'\b\d{9}\b', "US_SSN", ['ssn', 'social security'], 70),
        (r'\b\d{2}-\d{7}\b', "US_EIN", None, 80),
        (r'\b[1-2]\d{9}\b', "US_NPI", None, 75),
        (r'\b\d{5}(?:-\d{4})?\b', "US_ZIP", None, 60),
        (r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 90),
        (r'\+7\s*\d{3}\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 90),
        (r'8\s*\(\d{3}\)\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 85),
        (r'8\s*\d{3}\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 85),
        (r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', "PHONE", None, 70),
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "EMAIL", None, 90),
        (r'\b(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\b', "IP_ADDRESS", None, 85),
        (r'https?://[^\s/$.?#].[^\s]*', "URL", None, 75),
    ]
    
    COMPILED_PATTERNS = [(re.compile(p), et, ctx, pri) for p, et, ctx, pri in PATTERNS]
    FALSE_POSITIVE_SUBSTRINGS = ['example.com', 'localhost', '127.0.0.1', 'test', 'demo', 'sample', 'placeholder']
    
    def detect(self, text: str) -> list[dict]:
        matches = []
        for pattern, entity_type, context_keywords, priority in self.COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group()
                if any(fp in matched_text.lower() for fp in self.FALSE_POSITIVE_SUBSTRINGS):
                    continue
                if matched_text.startswith(' '):
                    continue
                if context_keywords:
                    ctx_start = max(0, match.start() - 30)
                    ctx_end = min(len(text), match.end() + 30)
                    context = text[ctx_start:ctx_end].lower()
                    if not any(kw in context for kw in context_keywords):
                        continue
                matches.append({"text": matched_text, "type": entity_type, "start": match.start(), "end": match.end(), "priority": priority, "source": "regex"})
        return self._resolve_overlaps(matches)
    
    def _resolve_overlaps(self, matches: list[dict]) -> list[dict]:
        if not matches:
            return []
        matches.sort(key=lambda x: (x['priority'], x['end'] - x['start']), reverse=True)
        resolved = []
        for match in matches:
            is_overlap = False
            for existing in resolved:
                if not (match['end'] <= existing['start'] or match['start'] >= existing['end']):
                    is_overlap = True
                    if match['text'] == existing['text']:
                        break
                    if len(match['text']) > len(existing['text']):
                        resolved.remove(existing)
                        is_overlap = False
                    break
            if not is_overlap:
                resolved.append(match)
        resolved.sort(key=lambda x: x['start'])
        return resolved


def merge_entities(regex_entities: list[dict], slm_entities: list[dict]) -> list[dict]:
    """Merge with deduplication."""
    merged = list(regex_entities)
    seen = {(e['text'].lower(), e['type']) for e in regex_entities}
    
    for ent in slm_entities:
        if not isinstance(ent, dict) or 'text' not in ent:
            continue
        ent_text_lower = ent['text'].lower()
        key = (ent_text_lower, ent.get('type', 'NAME'))
        if key in seen:
            continue
        is_overlap = False
        for re_ent in regex_entities:
            re_text_lower = re_ent['text'].lower()
            if ent_text_lower in re_text_lower or re_text_lower in ent_text_lower:
                is_overlap = True
                break
        if not is_overlap:
            seen.add(key)
            merged.append(ent)
    
    return merged


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_model(model: str, n_samples: int = 100) -> dict:
    """Benchmark a single model."""
    print(f"\n{'=' * 70}")
    print(f"TESTING: {model}")
    print("=" * 70)
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detector = RobustRegexDetector()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_regex_time = 0
    total_slm_time = 0
    errors = 0
    by_entity = {}
    
    for i, item in enumerate(data[:n_samples]):
        text = item.get('question', '')
        ground_truth = item.get('ground_truth', [])
        lang = item.get('language', 'ru')
        
        # Stage 1: Regex
        t0 = time.time()
        regex_ents = detector.detect(text)
        total_regex_time += time.time() - t0
        
        # Stage 2: SLM
        t0 = time.time()
        slm_ents = extract_with_json_schema(text, model)
        total_slm_time += time.time() - t0
        
        if not slm_ents and len(text) > 50:
            errors += 1
        
        # Merge
        merged = merge_entities(regex_ents, slm_ents)
        
        # Calculate metrics
        gt_texts = {e['text'] for e in ground_truth}
        found_texts = {e['text'] for e in merged}
        
        tp = len(gt_texts & found_texts)
        fp = len(found_texts - gt_texts)
        fn = len(gt_texts - found_texts)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Track by entity type
        for gt in ground_truth:
            etype = gt.get('type', 'UNKNOWN')
            if etype not in by_entity:
                by_entity[etype] = {'tp': 0, 'fp': 0, 'fn': 0}
            if gt['text'] in found_texts:
                by_entity[etype]['tp'] += 1
            else:
                by_entity[etype]['fn'] += 1
        
        for found in merged:
            if found['text'] not in gt_texts:
                etype = found.get('type', 'UNKNOWN')
                if etype not in by_entity:
                    by_entity[etype] = {'tp': 0, 'fp': 0, 'fn': 0}
                by_entity[etype]['fp'] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nRESULTS:")
    print(f"  Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    print(f"  Regex time: {total_regex_time:.2f}s ({total_regex_time/n_samples*1000:.1f}ms/sample)")
    print(f"  SLM time: {total_slm_time:.2f}s ({total_slm_time/n_samples:.2f}s/sample)")
    print(f"  Total time: {total_regex_time + total_slm_time:.2f}s ({(total_regex_time + total_slm_time)/n_samples:.2f}s/sample)")
    print(f"  Errors (empty responses): {errors}")
    
    # Entity-level breakdown
    print(f"\n  Entity-Level Breakdown:")
    for etype in sorted(by_entity.keys()):
        stats = by_entity[etype]
        e_tp = stats['tp']
        e_fp = stats['fp']
        e_fn = stats['fn']
        e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0
        e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0
        print(f"    {etype:<20} F1={e_f1:>5.1%} P={e_prec:>5.1%} R={e_rec:>5.1%} (TP={e_tp}, FP={e_fp}, FN={e_fn})")
    
    return {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "regex_time_ms": total_regex_time/n_samples*1000,
        "slm_time_s": total_slm_time/n_samples,
        "total_time_s": (total_regex_time + total_slm_time)/n_samples,
        "errors": errors,
        "by_entity": by_entity
    }


def run_full_benchmark(n_samples: int = 100):
    """Run full benchmark on all models."""
    print("=" * 70)
    print(f"COMPREHENSIVE BENCHMARK: 2-Stage SLM+REGEX Pipeline")
    print(f"Dataset: {n_samples} samples (EN + RU IT Support Tickets)")
    print("=" * 70)
    
    models = [
        "qwen/qwen3-8b",
        "qwen/qwen3-4b:free",
        "qwen/qwen3-1.7b:free",
        "openai/gpt-oss-20b",
        "google/gemma-3-4b-it",
    ]
    
    results = []
    for model in models:
        results.append(benchmark_model(model, n_samples))
    
    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time':>10} {'Errors':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<25} {r['f1']:>7.1%} {r['precision']:>9.1%} {r['recall']:>7.1%} {r['total_time_s']:>9.2f}s {r['errors']:>7}")
    print("=" * 70)
    
    # Find winner
    best = max(results, key=lambda x: x['f1'])
    print(f"\n🏆 WINNER: {best['model']} (F1: {best['f1']:.1%})")
    
    # Comparison to 3-Stage BERT
    print(f"\n{'=' * 70}")
    print("COMPARISON TO 3-STAGE BERT PIPELINE")
    print("=" * 70)
    print("3-Stage BERT (from plan v1.6, 200 samples):")
    print("  F1: 88.6%, Precision: 99.1%, Recall: 80.2%, Time: ~150ms")
    print(f"\n2-Stage SLM+REGEX (Qwen3-8B, {n_samples} samples):")
    print(f"  F1: {results[0]['f1']:.1%}, Precision: {results[0]['precision']:.1%}, Recall: {results[0]['recall']:.1%}, Time: {results[0]['total_time_s']:.2f}s")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_full_benchmark(100)
