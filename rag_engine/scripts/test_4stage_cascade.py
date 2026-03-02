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
# LANGUAGE DETECTION - Smart routing to appropriate NER
# =============================================================================

def detect_language(text):
    """Detect if text contains Russian (Cyrillic) or is primarily English."""
    # Count Cyrillic characters
    cyrillic_count = len(re.findall(r'[А-Яа-яЁё]', text))
    # Count ASCII letters
    latin_count = len(re.findall(r'[A-Za-z]', text))
    
    total = cyrillic_count + latin_count
    if total == 0:
        return 'en'  # Default to English
    
    cyrillic_ratio = cyrillic_count / total
    
    # If more than 30% Cyrillic, treat as Russian
    if cyrillic_ratio > 0.3:
        return 'ru'
    return 'en'


def smart_ner_stage(text, config):
    """Run appropriate NER based on language detection per sentence.
    
    For mixed EN+RU text, we detect language for each segment
    and route to the appropriate NER model.
    """
    config = config or 'smart_split'
    
    if config == 'smart_split':
        # Split into sentences/segments and detect language per segment
        import re
        segments = re.split(r'[,.\n]+', text)
        
        all_results = []
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            lang = detect_language(segment)
            
            if lang == 'ru':
                # Use Gherman for Russian
                all_results.extend(run_model(segment, 'gherman'))
            else:
                # Use dslim for English
                all_results.extend(run_model(segment, 'dslim'))
        
        return all_results
    
    elif config == 'gherman_only':
        return run_model(text, 'gherman')
    
    elif config == 'dslim_only':
        return run_model(text, 'dslim')
    
    else:
        # Default: dslim only
        return run_model(text, 'dslim')


# =============================================================================
# MERGE FUNCTION - Smart fragmentation handling
# =============================================================================

def is_cyrillic(s):
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r'[А-Яа-яЁё]', s))

def is_latin(s):
    """Check if text contains Latin characters."""
    return bool(re.search(r'[A-Za-z]', s))

def is_fragment(text, entity, all_text):
    """Check if entity is a fragmented piece of a larger word.
    
    Heuristics:
    - Single/two char fragment (BERT subword artifacts)
    - Wrong script for the context (e.g., Cyrillic 'т' in English-dominant text)
    - Low BERT confidence
    """
    clean = entity['text'].replace('##', '').strip()
    
    # ALWAYS filter single characters - almost always garbage
    if len(clean) <= 1:
        return True
    
    # Filter 2-char fragments with low confidence OR wrong script context
    if len(clean) <= 2:
        # Check confidence
        if entity.get('confidence', 1.0) < 0.85:
            return True
        
        # Check if it's in wrong-script context
        ctx_start = max(0, entity['start'] - 5)
        ctx_end = min(len(all_text), entity['end'] + 5)
        context = all_text[ctx_start:ctx_end]
        
        entity_is_cyrillic = is_cyrillic(clean)
        context_cyrillic = is_cyrillic(context.replace(clean, ''))
        context_latin = is_latin(context.replace(clean, ''))
        
        # If entity is Cyrillic but context is mostly Latin, it's garbage
        if entity_is_cyrillic and context_latin and not context_cyrillic:
            return True
        # If entity is Latin but context is mostly Cyrillic, it's garbage  
        if is_latin(clean) and context_cyrillic and not context_latin:
            return True
    
    return False


def merge_and_dedupe(all_entities, text):
    """
    Advanced merging with smart fragmentation handling:
    1. Filter obvious garbage (single-char, low confidence, wrong script)
    2. Cluster adjacent same-type entities (handles fragments and multi-word hits)
    3. Resolve overlaps (Longest Match First)
    4. Final cleanup
    """
    if not all_entities:
        return []
    
    # Pre-clean: Ensure text is synced with offsets
    for e in all_entities:
        e['text'] = text[e['start']:e['end']]
    
    # Step 1: Filter obvious garbage
    filtered = []
    for e in all_entities:
        if is_fragment(text, e, text):
            continue  # Skip garbage
        filtered.append(e)
    
    # Step 2: Sort by start position
    filtered.sort(key=lambda x: (x['start'], x['end']))
    
    # Step 3: Type grouping
    type_groups = {
        'LOCATION': 'LOC', 'ADDRESS': 'LOC', 'LOC': 'LOC',
        'PERSON': 'PER', 'NAME': 'PER', 'PER': 'PER',
        'ORGANIZATION': 'ORG', 'COMPANY': 'ORG', 'ORG': 'ORG',
        'PHONE': 'PHONE', 'EMAIL': 'EMAIL'
    }
    
    # Step 4: Cluster adjacent same-type entities
    clustered = []
    for e in filtered:
        if not clustered:
            clustered.append(e.copy())
            continue
        
        last = clustered[-1]
        
        group_last = type_groups.get(last['type'], last['type'])
        group_curr = type_groups.get(e['type'], e['type'])
        
        same_group = group_last == group_curr
        gap = e['start'] - last['end']
        
        # Check what separates the entities
        separator = text[last['end']:e['start']] if gap > 0 else ""
        surrounded_by_letters = (
            gap == 0 or  # Adjacent with no gap
            (gap > 0 and len(separator.strip()) == 0)  # Only punctuation/spaces
        )
        
        # Merge conditions:
        # 1. Same group AND
        # 2. (Surrounded by letters = fragmented word OR gap ≤ 3 for punctuation)
        if same_group and (surrounded_by_letters or gap <= 3):
            last['end'] = max(last['end'], e['end'])
            last['text'] = text[last['start']:last['end']]
            last['confidence'] = max(last['confidence'], e['confidence'])
            if e['source'] not in last['source']:
                last['source'] = f"{last['source']}+{e['source']}"
        else:
            clustered.append(e.copy())
            
    # Step 5: Resolve remaining overlaps (Longest Match First)
    clustered.sort(key=lambda x: (x['end'] - x['start'], -x['start']), reverse=True)
    
    covered = [False] * len(text)
    final = []
    
    for e in clustered:
        start, end = e['start'], e['end']
        if end <= start:
            continue
        
        overlaps = any(covered[i] for i in range(max(0, start), min(end, len(text))))
        
        if not overlaps:
            # Final cleanup: if too short and low confidence, skip
            clean = e['text'].replace('#', '').strip()
            if len(clean) < 2 and e.get('confidence', 1.0) < 0.8:
                continue
                
            final.append(e)
            for i in range(max(0, start), min(end, len(text))):
                covered[i] = True
    
    # Sort by position for final output
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
        
        # Smart NER: Language-aware routing
        if 'smart_ner' in stages:
            t0 = time.time()
            all_entities.extend(smart_ner_stage(sample, 'smart_split'))
            total_time += time.time() - t0
        
        # Stage 2: dslim (English) - only if not using smart_ner
        if 'dslim' in stages and 'smart_ner' not in stages:
            t0 = time.time()
            all_entities.extend(run_model(sample, 'dslim'))
            total_time += time.time() - t0
        
        # Stage 3: Gherman (Russian) - only if not using smart_ner
        if 'gherman' in stages and 'smart_ner' not in stages:
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
        (["regex", "dslim"], "2. Regex + dslim (EN only)"),
        (["regex", "gherman", "eu_pii"], "3. Regex + Gherman + EU-PII (RU biased)"),
        (["regex", "smart_ner", "eu_pii"], "4. SMART: Regex + LangDetect → dslim/Gherman + EU-PII"),
        (["regex", "dslim", "gherman", "eu_pii"], "5. FULL: Regex + dslim + Gherman + EU-PII"),
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
