"""
Multi-Model Benchmark: Compare SLM + REGEX across different models.

Models to test:
- openai/gpt-oss-20b
- qwen/qwen3-14b
- qwen/qwen3-8b
- qwen/qwen3-4b:free
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

MODELS = [
    "openai/gpt-oss-20b",  # Test GPT-OSS-20b
]


# =============================================================================
# REGEX DETECTOR (same for all models)
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
        (r'\bv?\d+\.\d+(\.\d+)?\b', "VERSION", None, 40),
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
                if entity_type == "VERSION":
                    if re.search(r'\d+\.\d+\.\d+\.\d+', matched_text):
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


# =============================================================================
# SLM EXTRACTOR
# =============================================================================

SLM_SYSTEM_PROMPT = """You are a PII extractor for IT Support Tickets.

Extract ONLY person names and organizations (companies).
- NAME: Person names (Иван Иванов, John Smith, Sarah Wilson, Петров В.А.)
- COMPANY: Organizations (Microsoft, Газпромнефть, Acme Corp)
- LOGIN: Usernames (admin, systemAccount)

Rules:
1. Extract full names, not partial (extract "John Smith", not just "John")
2. Skip technical terms (Kafka, Elasticsearch, SQL are NOT companies)
3. Skip email addresses (regex catches those)
4. Be conservative - only extract clear names/companies

Examples:
Input: "Contact Иван Иванов from Газпромнефть"
Output: [{"text": "Иван Иванов", "type": "NAME"}, {"text": "Газпромнефть", "type": "COMPANY"}]

Input: "John Smith reported the issue. Server 192.168.1.1"
Output: [{"text": "John Smith", "type": "NAME"}]

Return JSON array only.
Input: """


def slm_extract(text: str, model: str) -> list[dict]:
    """Use SLM to extract semantic entities."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SLM_SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            raw = re.sub(r'^```json?', '', raw)
            raw = re.sub(r'```$', '', raw)
        
        entities = json.loads(raw)
        return entities if isinstance(entities, list) else []
    except Exception as e:
        return []


def merge_entities(regex_entities: list[dict], slm_entities: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    
    for ent in regex_entities:
        key = (ent['text'].lower(), ent['type'])
        if key not in seen:
            seen.add(key)
            merged.append(ent)
    
    for ent in slm_entities:
        ent_text_lower = ent['text'].lower()
        key = (ent_text_lower, ent['type'])
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
    
    merged.sort(key=lambda x: x.get('start', 0))
    return merged


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_model(model: str, n_samples: int = 20) -> dict:
    """Benchmark a single model."""
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model}")
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
    
    for i, item in enumerate(data[:n_samples]):
        text = item.get('question', '')
        ground_truth = item.get('ground_truth', [])
        
        t0 = time.time()
        regex_ents = detector.detect(text)
        total_regex_time += time.time() - t0
        
        t0 = time.time()
        slm_ents = slm_extract(text, model)
        total_slm_time += time.time() - t0
        
        if not slm_ents and len(text) > 50:
            errors += 1
        
        merged = merge_entities(regex_ents, slm_ents)
        
        gt_texts = {e['text'] for e in ground_truth}
        found_texts = {e['text'] for e in merged}
        
        tp = len(gt_texts & found_texts)
        fp = len(found_texts - gt_texts)
        fn = len(gt_texts - found_texts)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    print(f"  Regex time: {total_regex_time:.2f}s ({total_regex_time/n_samples*1000:.1f}ms/sample)")
    print(f"  SLM time: {total_slm_time:.2f}s ({total_slm_time/n_samples:.2f}s/sample)")
    print(f"  Errors (empty response): {errors}")
    
    return {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "regex_time_ms": total_regex_time/n_samples*1000,
        "slm_time_s": total_slm_time/n_samples,
        "errors": errors
    }


def compare_models(n_samples: int = 20):
    """Compare all models."""
    print("=" * 70)
    print(f"MULTI-MODEL BENCHMARK ({n_samples} samples each)")
    print("=" * 70)
    
    results = []
    for model in MODELS:
        try:
            result = benchmark_model(model, n_samples)
            results.append(result)
        except Exception as e:
            print(f"ERROR with {model}: {e}")
            results.append({"model": model, "error": str(e)})
    
    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'SLM Time':>12}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<30} ERROR: {r['error'][:30]}")
        else:
            print(f"{r['model']:<30} {r['precision']:>9.1%} {r['recall']:>9.1%} {r['f1']:>9.1%} {r['slm_time_s']:>10.2f}s")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    compare_models(5)  # Test with 5 samples