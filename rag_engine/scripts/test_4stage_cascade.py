#!/usr/bin/env python3
"""
Test the recommended 4-stage cascade:
Stage 1: US + RU Regex
Stage 2: dslim_bert_large (English NER)
Stage 3: Gherman (Russian NER)
Stage 4: EU-PII (multilingual fallback)

Compare against alternatives.
"""

import time
import re
from dataclasses import dataclass
from typing import Any

# =============================================================================
# TEST DATA - Mixed RU/EN samples
# =============================================================================

MIXED_SAMPLES = [
    # English business
    "Please contact John Smith at john.smith@company.com or call +1-555-123-4567 for the project.",
    "Employee Maria Garcia (SSN: 123-45-6789) from Los Angeles, phone: +1-555-987-6543 needs access.",
    "Customer Robert Johnson - Email: r.johnson@customer.net, Phone: +1-555-111-2222, Address: 123 Main St, Boston MA 02101",
    
    # Russian business
    "Иван Иванов, +7-900-123-45-67, ivan.ivanov@example.com, Москва, ул. Ленина, д. 10, кв. 5. Работает менеджером в ООО 'Рога и Копыта'.",
    "Анна Смирнова, +7-901-234-56-78, anna.smirnova@company.ru, Санкт-Петербург, Невский пр-т, д. 25.",
    
    # Mixed content
    "Contact John Doe (john.doe@global.com, +1-555-000-1111) and Иван Петров (+7-902-333-44-55) for the project.",
    "Server at 192.168.1.100 with credentials user=admin, pass=Secret123. Russian contact: Алексей +7-903-444-55-66",
    
    # Medical/Financial
    "Patient Michael Brown, DOB: 03/15/1985, Medicare: 123456789012, email: mbrown@email.com",
    "Пациент Елена Козлова, полис ОМС: 1234567890123456, телефон: +7-904-555-66-77",
]

# =============================================================================
# COMBINED REGEX (US + RU)
# =============================================================================

class CombinedRegexPatterns:
    """Combined US + Russian regex patterns."""
    
    # Russian phones
    RU_PHONE_PATTERNS = [
        re.compile(r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'\+7\d{10}'),
        re.compile(r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'8\d{10}'),
    ]
    
    # US phones
    US_PHONE_PATTERNS = [
        re.compile(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    ]
    
    # Common patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    
    # Russian documents
    PASSPORT_PATTERN = re.compile(r'\b\d{2}\s+\d{2}\s+\d{6}\b|\b\d{4}\s+\d{6}\b')
    SNILS_PATTERN = re.compile(r'\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b')
    INN_PATTERN = re.compile(r'\b\d{10}\b|\b\d{12}\b')
    OGRN_PATTERN = re.compile(r'\b\d{13}\b|\b\d{15}\b')
    KPP_PATTERN = re.compile(r'\b\d{9}\b')
    
    # US documents
    US_SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    US_ITIN_PATTERN = re.compile(r'\b9\d{2}-\d{7}\b')
    US_EIN_PATTERN = re.compile(r'\b\d{2}-\d{7}\b')
    US_ZIP_PATTERN = re.compile(r'\b\d{5}(-\d{4})?\b')
    US_NPI_PATTERN = re.compile(r'\b[1-2]\d{9}\b')
    
    # Financial
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    
    # Corporate
    PASSWORD_PATTERN = re.compile(r'(пароль|password|passwd|pwd)[\s:=_]+(\S+)', re.IGNORECASE)
    USERNAME_PATTERN = re.compile(r'(username|user|login)[\s:=_]+(\S+)', re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|apikey)[\s:=_]+(\S+)', re.IGNORECASE)
    
    @classmethod
    def detect_all(cls, text: str) -> list[dict]:
        detections = []
        
        def add_detection(pattern, ptype, confidence=0.95):
            for match in pattern.finditer(text):
                overlaps = any(
                    not (match.end() <= d['start'] or match.start() >= d['end'])
                    for d in detections
                )
                if not overlaps:
                    detections.append({
                        'type': ptype,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': confidence,
                        'source': 'regex'
                    })
        
        # High-value patterns
        add_detection(cls.EMAIL_PATTERN, 'EMAIL')
        add_detection(cls.US_SSN_PATTERN, 'US_SSN')
        add_detection(cls.CREDIT_CARD_PATTERN, 'CREDIT_CARD')
        add_detection(cls.IP_PATTERN, 'IP')
        add_detection(cls.US_NPI_PATTERN, 'US_NPI')
        
        # Phones (all patterns)
        for p in cls.RU_PHONE_PATTERNS + cls.US_PHONE_PATTERNS:
            add_detection(p, 'PHONE')
        
        # Credentials
        add_detection(cls.PASSWORD_PATTERN, 'PASSWORD')
        add_detection(cls.USERNAME_PATTERN, 'USERNAME')
        add_detection(cls.API_KEY_PATTERN, 'API_KEY')
        
        # Russian docs (need context)
        add_detection(cls.SNILS_PATTERN, 'SNILS', ['снилс', 'инн'])
        add_detection(cls.PASSPORT_PATTERN, 'PASSPORT')
        add_detection(cls.INN_PATTERN, 'INN', ['инн', 'inn', 'налог'])
        add_detection(cls.OGRN_PATTERN, 'OGRN', ['огрн', 'ogrn'])
        
        # US patterns
        add_detection(cls.US_ZIP_PATTERN, 'US_ZIP')
        
        return sorted(detections, key=lambda x: x['start'])


# =============================================================================
# MODEL LOADING
# =============================================================================

_model_cache = {}

def load_model(name, **kwargs):
    if name in _model_cache:
        return _model_cache[name]
    try:
        from transformers import pipeline
        model = pipeline("ner", model=name, tokenizer=name, aggregation_strategy="simple", **kwargs)
        _model_cache[name] = model
        print(f"  Loaded: {name}")
        return model
    except Exception as e:
        print(f"  Failed: {name}: {e}")
        return None


MODELS = {
    'dslim': 'dslim/bert-large-NER',
    'gherman': 'Gherman/bert-base-NER-Russian',
    'eu_pii': 'tabularisai/eu-pii-safeguard',
}

ENTITY_MAPS = {
    'dslim': {'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'MISC': 'MISC'},
    'gherman': {'PER': 'PERSON', 'PERSON': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION'},
    'eu_pii': {'PERSON': 'PERSON', 'EMAIL_ADDRESS': 'EMAIL', 'PHONE_NUMBER': 'PHONE', 'LOCATION': 'LOCATION', 'ORGANIZATION': 'ORGANIZATION'},
}


def run_model(text, model_key):
    model = load_model(MODELS[model_key], device='cpu')
    if not model:
        return []
    
    results = model(text)
    detections = []
    label_map = ENTITY_MAPS.get(model_key, {})
    
    for r in results:
        label = r.get('entity_group', '')
        if label in label_map and r.get('score', 0) > 0.5:
            detections.append({
                'type': label_map[label],
                'text': r.get('word', ''),
                'start': r.get('start', 0),
                'end': r.get('end', 0),
                'confidence': r.get('score', 0),
                'source': model_key
            })
    
    return detections


# =============================================================================
# MERGE FUNCTION
# =============================================================================

def merge_and_dedupe(all_entities, text):
    if not all_entities:
        return []
    
    sorted_ents = sorted(all_entities, key=lambda x: (x['end'] - x['start'], -x['start']), reverse=True)
    
    covered = [False] * len(text)
    final = []
    
    for e in sorted_ents:
        start, end = e['start'], e['end']
        overlaps = any(covered[i] for i in range(start, min(end, len(text))))
        
        if not overlaps:
            final.append(e)
            for i in range(start, min(end, len(text))):
                covered[i] = True
    
    return sorted(final, key=lambda x: x['start'])


# =============================================================================
# TEST CASCADE
# =============================================================================

def test_cascade(stages, samples, name):
    total_time = 0
    total_detections = 0
    all_results = []
    
    for sample in samples:
        all_entities = []
        
        # Stage 1: Combined Regex
        if 'regex' in stages:
            t0 = time.time()
            all_entities.extend(CombinedRegexPatterns.detect_all(sample))
            total_time += time.time() - t0
        
        # Stage 2: dslim (English)
        if 'dslim' in stages:
            t0 = time.time()
            all_entities.extend(run_model(sample, 'dslim'))
            total_time += time.time() - t0
        
        # Stage 3: Gherman (Russian)
        if 'gherman' in stages:
            t0 = time.time()
            all_entities.extend(run_model(sample, 'gherman'))
            total_time += time.time() - t0
        
        # Stage 4: EU-PII
        if 'eu_pii' in stages:
            t0 = time.time()
            all_entities.extend(run_model(sample, 'eu_pii'))
            total_time += time.time() - t0
        
        merged = merge_and_dedupe(all_entities, sample)
        total_detections += len(merged)
        all_results.append(merged)
    
    return {
        'name': name,
        'stages': stages,
        'total_time': total_time,
        'avg_time_ms': (total_time / len(samples)) * 1000,
        'total_detections': total_detections,
        'avg_detections': total_detections / len(samples),
        'results': all_results
    }


def main():
    print("=" * 80)
    print("4-STAGE CASCADE TEST: US+RU Regex → dslim → Gherman → EU-PII")
    print("=" * 80)
    print(f"\nTesting {len(MIXED_SAMPLES)} mixed RU/EN samples\n")
    
    # Define cascades to test
    cascades = [
        (["regex"], "1. Regex Only (US+RU)"),
        (["regex", "dslim"], "2. Regex + dslim"),
        (["regex", "dslim", "gherman"], "3. Regex + dslim + Gherman"),
        (["regex", "dslim", "gherman", "eu_pii"], "4. FULL: Regex + dslim + Gherman + EU-PII"),
        (["regex", "dslim", "eu_pii"], "5. Regex + dslim + EU-PII"),
        (["regex", "gherman", "eu_pii"], "6. Regex + Gherman + EU-PII"),
    ]
    
    results = []
    
    for stages, name in cascades:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        r = test_cascade(stages, MIXED_SAMPLES, name)
        results.append(r)
        
        print(f"Time: {r['avg_time_ms']:.1f}ms/sample")
        print(f"Detections: {r['avg_detections']:.1f}/sample")
    
    # Show detailed comparison
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Cascade':<45} {'Time (ms)':<12} {'Dets':<8}")
    print("-" * 65)
    
    for r in sorted(results, key=lambda x: x['avg_time_ms']):
        print(f"{r['name']:<45} {r['avg_time_ms']:>8.1f}    {r['avg_detections']:>5.1f}")
    
    # Show detailed detections for the best cascade
    best = max(results, key=lambda x: x['avg_detections'])
    print(f"\n{'='*80}")
    print(f"DETAILED RESULTS: {best['name']}")
    print(f"{'='*80}")
    
    for i, sample in enumerate(MIXED_SAMPLES):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample[:70]}...")
        print("Detections:")
        for e in best['results'][i]:
            print(f"  [{e['source']:8s}] {e['type']:12s}: {e['text'][:40]}")


if __name__ == "__main__":
    main()
