"""
Combined SLM + REGEX Pipeline Test
Following the cascaded approach from anonymization plan:
1. Stage 1: REGEX (fast, structured IDs)
2. Stage 2: SLM (semantic - names, companies)
3. Deduplication & merge
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
# STAGE 1: REGEX PATTERNS (from anonymization plan)
# =============================================================================

class RegexDetector:
    """Fast regex-based PII detection - for structured IDs."""
    
    PATTERNS = {
        # Russian phones
        "PHONE": [
            re.compile(r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
            re.compile(r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
            re.compile(r'\+7\d{10}'),
            re.compile(r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
            re.compile(r'8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
            re.compile(r'8\d{10}'),
            re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),  # US/Intl
        ],
        "EMAIL": [re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')],
        "IP_ADDRESS": [re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')],
        "INN": [re.compile(r'\b\d{10}\b'), re.compile(r'\b\d{12}\b')],
        "PASSPORT": [re.compile(r'\b\d{4}\s*\d{6}\b')],
        "SNILS": [re.compile(r'\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b')],
        "OGRN": [re.compile(r'\b\d{13}\b'), re.compile(r'\b\d{15}\b')],
        "KPP": [re.compile(r'\b\d{9}\b')],
        "BIC": [re.compile(r'\b0[4-5]\d{7}\b')],
        "BANK_ACCOUNT": [re.compile(r'\b40[5-8]\d{17}\b'), re.compile(r'\b301\d{17}\b')],
        "BANK_CARD": [re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')],
        "US_SSN": [re.compile(r'\b\d{3}-\d{2}-\d{4}\b')],
        "US_ZIP": [re.compile(r'\b\d{5}(-\d{4})?\b')],
        "US_EIN": [re.compile(r'\b\d{2}-\d{7}\b')],
        "US_NPI": [re.compile(r'\b[1-2]\d{9}\b')],
        "URL": [re.compile(r'https?://[^\s/$.?#].[^\s]*')],
        "VERSION": [re.compile(r'\b\d+\.\d+(\.\d+)?\b')],
    }
    
    CONTEXT_KEYWORDS = {
        "INN": ['инн', 'inn', 'налог'],
        "OGRN": ['огрн', 'ogrn'],
        "KPP": ['кпп', 'kpp'],
        "BIC": ['бик', 'bic'],
    }
    
    def detect(self, text: str) -> list[dict]:
        """Detect structured PII using regex."""
        entities = []
        seen = set()
        
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Check for context keywords
                    if entity_type in self.CONTEXT_KEYWORDS:
                        ctx_start = max(0, match.start() - 20)
                        context = text[ctx_start:match.end()].lower()
                        if not any(kw in context for kw in self.CONTEXT_KEYWORDS[entity_type]):
                            continue
                    
                    # Deduplicate
                    key = (match.group(), entity_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    entities.append({
                        "text": match.group(),
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "source": "regex"
                    })
        
        return entities


# =============================================================================
# STAGE 2: SLM (for names, companies - semantic understanding)
# =============================================================================

SYSTEM_PROMPT = """You are a PII extractor. Extract names and organizations.

Entity types: NAME, COMPANY, LOGIN

Use regex to find patterns:
- Names: capitalized words (Иван Иванов, John Smith)
- Companies: organizations (Microsoft, Газпромнефть, Acme Corp)
- Logins: admin, systemAccount, service_account

Examples:
Input: "Иван Иванов из Газпромнефть"
Output: [{"text": "Иван Иванов", "type": "NAME"}, {"text": "Газпромнефть", "type": "COMPANY"}]

Input: "Contact John Smith at john.smith@acme.com"
Output: [{"text": "John Smith", "type": "NAME"}, {"text": "acme.com", "type": "COMPANY"}]

Input: "systemAccount is locked"
Output: [{"text": "systemAccount", "type": "LOGIN"}]

Return JSON array only. No markdown.
Input: """


def slm_extract(text: str) -> list[dict]:
    """Use SLM to extract semantic entities (names, companies)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        
        # Clean markdown
        if raw.startswith("```"):
            raw = re.sub(r'^```json?', '', raw)
            raw = re.sub(r'```$', '', raw)
        
        entities = json.loads(raw)
        return entities if isinstance(entities, list) else []
    except Exception as e:
        print(f"SLM Error: {e}")
        return []


# =============================================================================
# STAGE 3: DEDUPLICATION & MERGE
# =============================================================================

def merge_entities(regex_entities: list[dict], slm_entities: list[dict]) -> list[dict]:
    """Merge regex + SLM entities with deduplication."""
    merged = []
    seen = set()  # (text_lower, type)
    
    # Add regex entities first (higher confidence for structured IDs)
    for ent in regex_entities:
        key = (ent['text'].lower(), ent['type'])
        if key not in seen:
            seen.add(key)
            merged.append(ent)
    
    # Add SLM entities (skip if already covered by regex)
    for ent in slm_entities:
        key = (ent['text'].lower(), ent['type'])
        if key not in seen:
            # Check for overlap with regex
            is_overlap = False
            for re_ent in regex_entities:
                if ent['text'].lower() in re_ent['text'].lower() or re_ent['text'].lower() in ent['text'].lower():
                    is_overlap = True
                    break
            if not is_overlap:
                seen.add(key)
                merged.append(ent)
    
    return merged


def anonymize_text(text: str, entities: list[dict]) -> tuple[str, dict]:
    """Apply anonymization to text."""
    mapping = {}
    
    # Sort by length (longest first)
    sorted_ents = sorted(entities, key=lambda x: len(x['text']), reverse=True)
    
    anonymized = text
    counter = 1
    
    for ent in sorted_ents:
        entity_text = ent['text']
        entity_type = ent['type']
        
        if entity_text in anonymized:
            placeholder = f"[{entity_type}_{counter}]"
            mapping[placeholder] = entity_text
            anonymized = anonymized.replace(entity_text, placeholder)
            counter += 1
    
    return anonymized, mapping


# =============================================================================
# TEST
# =============================================================================

def test_pipeline():
    """Test the combined SLM + REGEX pipeline."""
    print("=" * 70)
    print("COMBINED SLM + REGEX PIPELINE TEST")
    print("=" * 70)
    
    regex_detector = RegexDetector()
    
    test_cases = [
        "Добрый день! Пользователь Иван Иванов из Газпромнефть. Email: ivanov@gazpromneft.ru, тел: +7 495 123-45-67. Сервер 192.168.1.10. Версия 5.0.13334.0. ИНН: 7714012345.",
        "Contact John Smith at john.smith@acme.com. Server 10.0.0.50. US SSN: 123-45-6789. ZIP: 90210. Version 2.1.0",
        "Учетная запись systemAccount заблокирована. ОГРН: 1167746071421. БИК: 044525201.",
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {text[:80]}...")
        
        # Stage 1: Regex
        t0 = time.time()
        regex_entities = regex_detector.detect(text)
        t1 = time.time()
        print(f"REGEX ({t1-t0:.3f}s): {regex_entities}")
        
        # Stage 2: SLM
        t0 = time.time()
        slm_entities = slm_extract(text)
        t1 = time.time()
        print(f"SLM ({t1-t0:.3f}s): {slm_entities}")
        
        # Stage 3: Merge
        merged = merge_entities(regex_entities, slm_entities)
        print(f"MERGED: {merged}")
        
        # Anonymize
        anon_text, mapping = anonymize_text(text, merged)
        print(f"ANONYMIZED: {anon_text[:80]}...")
        print(f"MAPPING: {mapping}")


def benchmark(n_samples: int = 5):
    """Run on enriched dataset."""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {n_samples} samples")
    print("=" * 70)
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    regex_detector = RegexDetector()
    
    total_regex = 0
    total_slm = 0
    total_merged = 0
    
    for i, item in enumerate(data[:n_samples]):
        text = item.get('question', '')
        ground_truth = item.get('ground_truth', [])
        
        # Run pipeline
        regex_entities = regex_detector.detect(text)
        slm_entities = slm_extract(text)
        merged = merge_entities(regex_entities, slm_entities)
        
        # Calculate metrics
        gt_texts = {e['text'] for e in ground_truth}
        found_texts = {e['text'] for e in merged}
        
        tp = len(gt_texts & found_texts)
        fp = len(found_texts - gt_texts)
        fn = len(gt_texts - found_texts)
        
        print(f"\nSample {i+1}: {len(merged)} found, {len(ground_truth)} GT")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        if fp:
            print(f"  False Pos: {found_texts - gt_texts}")
        
        total_regex += len(regex_entities)
        total_slm += len(slm_entities)
        total_merged += len(merged)
    
    print(f"\n--- Summary ---")
    print(f"Avg regex: {total_regex/n_samples:.1f}")
    print(f"Avg SLM: {total_slm/n_samples:.1f}")
    print(f"Avg merged: {total_merged/n_samples:.1f}")


if __name__ == "__main__":
    test_pipeline()
    # benchmark(5)  # Too slow for API