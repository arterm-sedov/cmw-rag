"""
Robust SLM + REGEX Pipeline for PII Detection
Based on:
- .opencode/plans/20260227-anonymization-implementation-plan.md
- Microsoft Presidio patterns
- rus-anonymizer, ru-smb-pd-anonymizer

Architecture:
1. Stage 1: Context-aware REGEX (structured IDs - fast, high precision)
2. Stage 2: SLM (semantic - names, companies - context understanding)
3. Stage 3: Deduplication & merge (longest-match-first)
"""

import os
import json
import re
import time
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("SLM_MODEL", "openai/gpt-oss-20b")
DATASET_PATH = r"D:\Repo\cmw-rag\rag_engine\data\synthetic_cmw_support_tickets_v1_with_pii.json"

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# =============================================================================
# STAGE 1: ROBUST REGEX DETECTOR
# Based on anonymization plan + Presidio + rus-anonymizer
# =============================================================================

class RobustRegexDetector:
    """
    Context-aware regex detection with:
    - Priority-based matching (longer/more specific patterns first)
    - Context keyword validation for ambiguous patterns
    - Overlap resolution (longest match wins)
    - Proper deduplication
    """
    
    # Priority: higher = more specific = check first
    # Format: (pattern, entity_type, context_keywords, priority)
    PATTERNS = [
        # High priority - specific patterns with context
        (r'(?:пароль|password|passwd|pwd)[\s:=_]+(\S+)', "PASSWORD", None, 100),
        (r'(?:логин|login|username)[\s:=_]+(\S+)', "LOGIN", None, 100),
        (r'(?:api[_-]?key|apikey)[\s:=_]+(\S+)', "API_KEY", None, 100),
        
        # High priority - Russian docs with context
        (r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}\b', "SNILS", ['снилс', 'snils'], 90),
        (r'\b\d{10}\b', "INN", ['инн', 'inn', 'налог'], 85),
        (r'\b\d{12}\b', "INN", ['инн', 'inn', 'налог'], 85),
        (r'\b\d{13}\b', "OGRN", ['огрн', 'ogrn'], 85),
        (r'\b\d{15}\b', "OGRN", ['огрн', 'ogrn'], 85),
        (r'\b\d{9}\b', "KPP", ['кпп', 'kpp'], 80),
        
        # High priority - Russian passport (4+6 digits)
        (r'\b\d{4}\s*\d{6}\b', "PASSPORT", None, 85),
        
        # High priority - Bank/Financial
        (r'\b0[4-5]\d{7}\b', "BIC", ['бик', 'bic'], 85),
        (r'\b40[5-8]\d{17}\b', "BANK_ACCOUNT", None, 90),
        (r'\b301\d{17}\b', "BANK_ACCOUNT", None, 90),
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "BANK_CARD", None, 80),
        
        # High priority - US SSN
        (r'\b\d{3}-\d{2}-\d{4}\b', "US_SSN", None, 95),
        (r'\b\d{9}\b', "US_SSN", ['ssn', 'social security'], 70),
        
        # Medium priority - US IDs
        (r'\b\d{2}-\d{7}\b', "US_EIN", None, 80),
        (r'\b[1-2]\d{9}\b', "US_NPI", None, 75),
        (r'\b\d{5}(?:-\d{4})?\b', "US_ZIP", None, 60),  # Lower - too common
        
        # High priority - Phone (specific formats)
        (r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 90),
        (r'\+7\s*\d{3}\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 90),
        (r'8\s*\(\d{3}\)\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 85),
        (r'8\s*\d{3}\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}', "PHONE", None, 85),
        (r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', "PHONE", None, 70),
        
        # High priority - Email
        (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "EMAIL", None, 90),
        
        # High priority - IP Address (strict IPv4)
        (r'\b(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\.(?:25[0-5]|2[0-4]\d|1?\d?\d)\b', "IP_ADDRESS", None, 85),
        
        # High priority - URLs
        (r'https?://[^\s/$.?#].[^\s]*', "URL", None, 75),
        
        # Version numbers (lower priority - often false positives)
        (r'\bv?\d+\.\d+(\.\d+)?\b', "VERSION", None, 40),
    ]
    
    # Compile patterns
    COMPILED_PATTERNS = [(re.compile(p), et, ctx, pri) for p, et, ctx, pri in PATTERNS]
    
    # Known false positives to filter
    FALSE_POSITIVE_SUBSTRINGS = [
        'example.com', 'localhost', '127.0.0.1',
        'test', 'demo', 'sample', 'placeholder',
    ]
    
    def detect(self, text: str) -> list[dict]:
        """Detect PII using context-aware regex."""
        matches = []
        
        for pattern, entity_type, context_keywords, priority in self.COMPILED_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group()
                
                # Check for false positive substrings
                if any(fp in matched_text.lower() for fp in self.FALSE_POSITIVE_SUBSTRINGS):
                    continue
                
                # Skip matches starting with space (partial matches)
                if matched_text.startswith(' '):
                    continue
                
                # Skip version-like patterns that overlap with IP
                if entity_type == "VERSION":
                    if re.search(r'\d+\.\d+\.\d+\.\d+', matched_text):
                        continue
                
                # Check context keywords if required
                if context_keywords:
                    ctx_start = max(0, match.start() - 30)
                    ctx_end = min(len(text), match.end() + 30)
                    context = text[ctx_start:ctx_end].lower()
                    if not any(kw in context for kw in context_keywords):
                        continue
                
                matches.append({
                    "text": matched_text,
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "priority": priority,
                    "source": "regex"
                })
        
        # Resolve overlaps: keep longest/highest priority
        return self._resolve_overlaps(matches)
    
    def _resolve_overlaps(self, matches: list[dict]) -> list[dict]:
        """Keep longest match when overlaps occur."""
        if not matches:
            return []
        
        # Sort by priority (desc) then by length (desc)
        matches.sort(key=lambda x: (x['priority'], x['end'] - x['start']), reverse=True)
        
        resolved = []
        for match in matches:
            is_overlap = False
            for existing in resolved:
                # Check overlap
                if not (match['end'] <= existing['start'] or match['start'] >= existing['end']):
                    is_overlap = True
                    # If same text, prefer existing
                    if match['text'] == existing['text']:
                        break
                    # If different, prefer longer
                    if len(match['text']) > len(existing['text']):
                        resolved.remove(existing)
                        is_overlap = False
                    break
            if not is_overlap:
                resolved.append(match)
        
        # Sort by position
        resolved.sort(key=lambda x: x['start'])
        return resolved


# =============================================================================
# STAGE 2: SLM EXTRACTOR
# For semantic entities (names, companies) that regex can't detect
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


def slm_extract(text: str) -> list[dict]:
    """Use SLM to extract semantic entities."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
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
        print(f"SLM Error: {e}")
        return []


# =============================================================================
# STAGE 3: MERGE & DEDUPLICATE
# =============================================================================

def merge_entities(regex_entities: list[dict], slm_entities: list[dict], text: str) -> list[dict]:
    """Merge regex + SLM with proper deduplication."""
    merged = []
    seen = set()  # (text_lower, type)
    
    # Add regex entities first (higher confidence for structured IDs)
    for ent in regex_entities:
        key = (ent['text'].lower(), ent['type'])
        if key not in seen:
            seen.add(key)
            merged.append(ent)
    
    # Add SLM entities (check for overlap with regex)
    for ent in slm_entities:
        ent_text_lower = ent['text'].lower()
        key = (ent_text_lower, ent['type'])
        
        # Skip if already covered by regex
        if key in seen:
            continue
        
        # Check for text overlap
        is_overlap = False
        for re_ent in regex_entities:
            re_text_lower = re_ent['text'].lower()
            # Skip if regex already captured this text
            if ent_text_lower in re_text_lower or re_text_lower in ent_text_lower:
                is_overlap = True
                break
        
        if not is_overlap:
            seen.add(key)
            merged.append(ent)
    
    # Sort by position
    merged.sort(key=lambda x: x.get('start', 0))
    return merged


def anonymize_text(text: str, entities: list[dict]) -> tuple[str, dict]:
    """Apply anonymization to text."""
    mapping = {}
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_ents = sorted(entities, key=lambda x: len(x['text']), reverse=True)
    
    anonymized = text
    counter = {}
    
    for ent in sorted_ents:
        entity_text = ent['text']
        entity_type = ent['type']
        
        if entity_text in anonymized:
            # Get or create counter for this type
            if entity_type not in counter:
                counter[entity_type] = 1
            else:
                counter[entity_type] += 1
            
            placeholder = f"[{entity_type}_{counter[entity_type]}]"
            mapping[placeholder] = entity_text
            anonymized = anonymized.replace(entity_text, placeholder)
    
    return anonymized, mapping


# =============================================================================
# TEST & BENCHMARK
# =============================================================================

def test_pipeline():
    """Test the robust pipeline."""
    print("=" * 70)
    print("ROBUST SLM + REGEX PIPELINE TEST")
    print("=" * 70)
    
    detector = RobustRegexDetector()
    
    test_cases = [
        "Добрый день! Пользователь Иван Иванов из Газпромнефть. Email: ivanov@gazpromneft.ru, тел: +7 495 123-45-67. Сервер 192.168.1.10. Версия 5.0.13334.0. ИНН: 7714012345.",
        "Contact John Smith at john.smith@acme.com. Server 10.0.0.50. US SSN: 123-45-6789. ZIP: 90210.",
        "Учетная запись systemAccount заблокирована. ОГРН: 1167746071421. БИК: 044525201.",
        # False positive test
        "В кластере Kafka накопилось много данных. Используем Elasticsearch для логов.",
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {text[:80]}...")
        
        t0 = time.time()
        regex_ents = detector.detect(text)
        t1 = time.time()
        
        t2 = time.time()
        slm_ents = slm_extract(text)
        t3 = time.time()
        
        merged = merge_entities(regex_ents, slm_ents, text)
        
        print(f"REGEX ({t1-t0:.3f}s): {[(e['text'], e['type']) for e in regex_ents]}")
        print(f"SLM ({t3-t2:.2f}s): {slm_ents}")
        print(f"MERGED: {[(e['text'], e['type']) for e in merged]}")
        
        anon_text, mapping = anonymize_text(text, merged)
        print(f"ANONYMIZED: {anon_text[:80]}...")


def benchmark(n_samples: int = 50):
    """Benchmark on enriched dataset."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {n_samples} samples")
    print("=" * 70)
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detector = RobustRegexDetector()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_regex_time = 0
    total_slm_time = 0
    
    for i, item in enumerate(data[:n_samples]):
        text = item.get('question', '')
        ground_truth = item.get('ground_truth', [])
        
        # Run pipeline
        t0 = time.time()
        regex_ents = detector.detect(text)
        total_regex_time += time.time() - t0
        
        t0 = time.time()
        slm_ents = slm_extract(text)
        total_slm_time += time.time() - t0
        
        merged = merge_entities(regex_ents, slm_ents, text)
        
        # Calculate metrics
        gt_texts = {e['text'] for e in ground_truth}
        found_texts = {e['text'] for e in merged}
        
        tp = len(gt_texts & found_texts)
        fp = len(found_texts - gt_texts)
        fn = len(gt_texts - found_texts)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if i < 5:
            print(f"\nSample {i+1}: TP={tp}, FP={fp}, FN={fn}")
            if fp:
                print(f"  FP: {found_texts - gt_texts}")
            if fn:
                print(f"  FN: {gt_texts - found_texts}")
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"RESULTS:")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    print(f"  Regex time: {total_regex_time:.2f}s ({total_regex_time/n_samples*1000:.1f}ms/sample)")
    print(f"  SLM time: {total_slm_time:.2f}s ({total_slm_time/n_samples:.2f}s/sample)")
    print("=" * 70)
    
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    # test_pipeline()  # Skip for faster benchmark
    benchmark(50)  # Run benchmark on 50 samples