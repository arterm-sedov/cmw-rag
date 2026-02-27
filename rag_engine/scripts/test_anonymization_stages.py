#!/usr/bin/env python3
"""
Ad-hoc test script for anonymization pipeline stages.
Tests each stage on synthetic Russian PII data.
"""

import time
import re
import yaml
from pathlib import Path
from typing import Any

# =============================================================================
# TEST DATA - Synthetic Russian PII samples from notebook
# =============================================================================

RUSSIAN_SAMPLES = [
    "Иван Иванов, +7-900-123-45-67, ivan.ivanov@example.com, Москва, ул. Ленина, д. 10, кв. 5. Работает менеджером в ООО 'Рога и Копыта'.",
    "Анна Смирнова, +7-901-234-56-78, anna.smirnova@company.ru, Санкт-Петербург, Невский пр-т, д. 25, оф. 10. Является директором в 'ТехноПрогресс'.",
    "Дмитрий Петров, +7-902-345-67-89, dmitry.petrov@mail.org, Новосибирск, ул. Кирова, д. 3, кв. 1. Инженер-программист в АО 'СибСофт'.",
    "Елена Козлова, +7-903-456-78-90, elena.kozlova@web.net, Екатеринбург, пр-т Ленина, д. 15, кв. 8. Занимает должность главного бухгалтера в ПАО 'УралФинанс'.",
    "Иван Морозов, аналитик в ИП 'Морозов Консалтинг', телефон для связи: +7-909-333-44-55, i.morozov@morozov-consulting.ru, фактический адрес: г. Казань, ул. Пушкина, д. 15.",
]

CORPORATE_SAMPLES = [
    "Дмитрий Смирнов, директор по продажам в ООО 'Технострой', тел: +7-905-111-22-33, dmitry.smirnov@tekhnostroy.ru, адрес: г. Санкт-Петербург, Невский пр-т, д. 100.",
    "Ольга Иванова, руководитель отдела маркетинга АО 'ЭнергоПром', мобильный: +7-906-444-55-66, olga.ivanova@energoprom.org, офис: г. Москва, ул. Тверская, д. 25.",
    "Алексей Петров, главный инженер ЗАО 'СтройИнвест', контактный телефон: +7-907-777-88-99, a.petrov@stroyinvest.net, склад: г. Екатеринбург, пр-т Космонавтов, д. 30.",
    "Екатерина Васильева, менеджер по персоналу ПАО 'ФинансГрупп', раб. тел.: +7-908-000-11-22, e.vasilyeva@financegroup.com, центральный офис: г. Новосибирск, ул. Ленина, д. 5.",
]

# Additional edge cases
EDGE_CASES = [
    # Russian documents
    "Паспорт: 45 77 123456, ИНН: 1234567890, СНИЛС: 123-456-789 00",
    "ОГРН: 1234567890123, КПП: 123456789, ОКПО: 12345678, ОКВЭД: 62.01",
    "БИК: 044525700, р/с: 40702810380000001234, к/с: 30101810400000000700",
    
    # Vehicle
    "Госномер: А777АА77, VIN: WA1LAAFPNKD123456",
    
    # Corporate sensitive
    "Пароль: SuperSecret123, логин: admin, API ключ: sk_live_abc123xyz",
    "Внутренний ресурс: http://intranet.company.local/dashboard, IP: 192.168.1.105",
    "Публичный URL: https://github.com/comindware/cmw-rag",
    
    # Age & Dates
    "Мне 35 лет, родился 15.03.1990. Сегодня 27 февраля 2026 года.",
    
    # Financial
    "Номер карты: 4276 1234 5678 9012, срок: 12/28",
    
    # Additional Russian IDs
    "В/у: 77 12 123456, Полис ОМС: 1234567890123456",
    "Св-во о рождении: I-АВ № 123456, Военный билет: АБ № 1234567",
    
    # Location details
    "Индекс: 101000, Страна: Россия, Регион: Московская область",
]

ALL_SAMPLES = RUSSIAN_SAMPLES + CORPORATE_SAMPLES + EDGE_CASES


# =============================================================================
# STAGE 1: REGEX PATTERNS (Simplified - matches our plan)
# =============================================================================

class RegexPatterns:
    """Simplified regex patterns matching our Stage 1 plan."""
    
    # Russian phone patterns (6 formats from rus-anonymizer)
    # Fixed: Added dash support without spaces, and word boundaries
    PHONE_PATTERNS = [
        re.compile(r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),  # +7 (XXX) XXX-XX-XX
        re.compile(r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),   # +7 XXX XXX-XX-XX
        re.compile(r'\+7\d{10}'),                                         # +7XXXXXXXXXX
        re.compile(r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),   # 8 (XXX) XXX-XX-XX
        re.compile(r'8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),     # 8 XXX XXX-XX-XX
        re.compile(r'8\d{10}'),                                             # 8XXXXXXXXXX
        # Additional patterns for common Russian phone formats
        re.compile(r'\+7-\d{3}-\d{3}-\d{2}-\d{2}'),  # +7-XXX-XXX-XX-XX
        re.compile(r'8-\d{3}-\d{3}-\d{2}-\d{2}'),    # 8-XXX-XXX-XX-XX
    ]
    
    # Email
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    
    # Russian documents
    PASSPORT_PATTERN = re.compile(r'\b\d{2}\s+\d{2}\s+\d{6}\b|\b\d{4}\s+\d{6}\b')
    SNILS_PATTERN = re.compile(r'\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b')
    INN_PATTERN = re.compile(r'\b\d{10}\b|\b\d{12}\b')
    OGRN_PATTERN = re.compile(r'\b\d{13}\b|\b\d{15}\b')
    KPP_PATTERN = re.compile(r'\b\d{9}\b')
    OKPO_PATTERN = re.compile(r'\b\d{8}\b|\b\d{10}\b')
    OKVED_PATTERN = re.compile(r'\b\d{2}\.\d{2}(\.\d{1,2})?\b')
    
    # Financial
    BIC_PATTERN = re.compile(r'\b0[4-5]\d{7}\b')
    BANK_ACCOUNT_PATTERN = re.compile(r'\b40[5-8]\d{17}\b|\b301\d{17}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    
    # Vehicle
    CAR_NUMBER_PATTERN = re.compile(r'\b[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}\b')
    VIN_PATTERN = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')
    
    # Sensitive
    PASSWORD_PATTERN = re.compile(r'(пароль|password|passwd|pwd)[_\s:=]+[^\s]+', re.IGNORECASE)
    LOGIN_PATTERN = re.compile(r'(логин|login|username)[_\s:=]+[^\s]+', re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|apikey|ключ)[_\s:=]+[^\s]+', re.IGNORECASE)
    INTERNAL_URL_PATTERN = re.compile(r'(http|https)://(intranet|internal|local|192\.168|10\.|172\.(1[6-9]|2|3[01]))[^\s,]*', re.IGNORECASE)
    URL_PATTERN = re.compile(r'https?://[^\s/$.?#].[^\s,]*', re.IGNORECASE)
    IP_ADDRESS_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    
    # Age & Dates
    AGE_PATTERN = re.compile(r'мне\s+\d{1,2}\s+лет', re.IGNORECASE)
    DATE_PATTERN = re.compile(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{2,4}\b', re.IGNORECASE)
    
    # Additional Russian IDs
    DRIVER_LICENSE_PATTERN = re.compile(r'\b\d{2}\s*\d{2}\s*\d{6}\b')
    OMS_POLICY_PATTERN = re.compile(r'\b\d{16}\b')
    BIRTH_CERT_PATTERN = re.compile(r'\b[IVXLCDM]+-[А-Я]{2}\s+№?\s*\d{6}\b')
    MILITARY_ID_PATTERN = re.compile(r'\b[А-Я]{2}\s+№?\s*\d{7}\b')
    
    # Location
    POSTAL_CODE_PATTERN = re.compile(r'\b\d{6}\b')
    
    @classmethod
    def _has_overlap(cls, start: int, end: int, detections: list[dict]) -> bool:
        """Check if a range overlaps with any existing detection."""
        for d in detections:
            if not (end <= d['start'] or start >= d['end']):
                return True
        return False
    
    @classmethod
    def detect_all(cls, text: str) -> list[dict[str, Any]]:
        """Detect all PII using regex patterns with proper overlap handling."""
        detections = []
        
        # Helper to add detection if no overlap
        def add_if_no_overlap(pattern, type_name, context_keywords=None):
            for match in pattern.finditer(text):
                if not cls._has_overlap(match.start(), match.end(), detections):
                    if context_keywords:
                        ctx_start = max(0, match.start() - 20)
                        context = text[ctx_start:match.end()].lower()
                        if not any(kw in context for kw in context_keywords):
                            continue
                    detections.append({
                        "type": type_name, "text": match.group(),
                        "start": match.start(), "end": match.end(),
                    })

        # Priority 1: Specific Long IDs (no context needed)
        add_if_no_overlap(cls.BANK_ACCOUNT_PATTERN, "BANK_ACCOUNT")
        add_if_no_overlap(cls.VIN_PATTERN, "VIN")
        add_if_no_overlap(cls.EMAIL_PATTERN, "EMAIL")
        add_if_no_overlap(cls.OMS_POLICY_PATTERN, "OMS_POLICY") # 16 digits
        
        # Priority 2: Patterns with specific markers (no context needed)
        add_if_no_overlap(cls.PASSWORD_PATTERN, "PASSWORD")
        add_if_no_overlap(cls.LOGIN_PATTERN, "LOGIN")
        add_if_no_overlap(cls.API_KEY_PATTERN, "API_KEY")
        add_if_no_overlap(cls.INTERNAL_URL_PATTERN, "INTERNAL_URL")
        
        # Priority 3: Phone & Cards (strong patterns)
        for pattern in cls.PHONE_PATTERNS:
            add_if_no_overlap(pattern, "PHONE")
        add_if_no_overlap(cls.CREDIT_CARD_PATTERN, "CREDIT_CARD")
        
        # Priority 4: Russian IDs (may need context or have distinct format)
        add_if_no_overlap(cls.SNILS_PATTERN, "SNILS")
        add_if_no_overlap(cls.PASSPORT_PATTERN, "PASSPORT")
        add_if_no_overlap(cls.CAR_NUMBER_PATTERN, "CAR_NUMBER")
        add_if_no_overlap(cls.BIRTH_CERT_PATTERN, "BIRTH_CERT")
        add_if_no_overlap(cls.MILITARY_ID_PATTERN, "MILITARY_ID")
        
        # Priority 5: IDs that NEED context (numeric strings)
        add_if_no_overlap(cls.INN_PATTERN, "INN", ['инн', 'inn', 'налог'])
        add_if_no_overlap(cls.OGRN_PATTERN, "OGRN", ['огрн', 'ogrn'])
        add_if_no_overlap(cls.KPP_PATTERN, "KPP", ['кпп', 'kpp'])
        add_if_no_overlap(cls.BIC_PATTERN, "BIC", ['бик', 'bic'])
        add_if_no_overlap(cls.OKPO_PATTERN, "OKPO", ['окпо', 'okpo'])
        add_if_no_overlap(cls.DRIVER_LICENSE_PATTERN, "DRIVER_LICENSE", ['в/у', 'права', 'удостоверение'])
        add_if_no_overlap(cls.OKVED_PATTERN, "OKVED", ['оквэд', 'okved', 'деятельности'])
        add_if_no_overlap(cls.POSTAL_CODE_PATTERN, "POSTAL_CODE", ['индекс', 'index', 'почта', '101000'])
        
        # Priority 6: General patterns (last)
        add_if_no_overlap(cls.URL_PATTERN, "URL")
        add_if_no_overlap(cls.IP_ADDRESS_PATTERN, "IP_ADDRESS")
        add_if_no_overlap(cls.AGE_PATTERN, "AGE")
        add_if_no_overlap(cls.DATE_PATTERN, "DATE_TIME")
        
        # Sort by position
        detections.sort(key=lambda x: x['start'])
        return detections


def test_stage1_regex():
    """Test Stage 1: Regex patterns."""
    print("\n" + "=" * 80)
    print("STAGE 1: REGEX PATTERNS")
    print("=" * 80)
    
    total_detections = 0
    total_time = 0
    
    for i, sample in enumerate(ALL_SAMPLES):
        start_time = time.time()
        detections = RegexPatterns.detect_all(sample)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Original: {sample[:80]}...")
        print(f"Time: {elapsed*1000:.1f}ms")
        
        if detections:
            total_detections += len(detections)
            print(f"Detected ({len(detections)}):")
            for d in detections:
                print(f"  - {d['type']}: {d['text']}")
        else:
            print("  No PII detected")
    
    print(f"\n=== Stage 1 Summary ===")
    print(f"Total samples: {len(ALL_SAMPLES)}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {total_time*1000:.1f}ms")
    print(f"Avg per sample: {total_time/len(ALL_SAMPLES)*1000:.1f}ms")


# =============================================================================
# STAGE 2: EU-PII-SAFEGUARD (if available)
# =============================================================================

def test_stage2_eu_pii():
    """Test Stage 2: EU-PII-Safeguard model."""
    print("\n" + "=" * 80)
    print("STAGE 2: EU-PII-SAFEGUARD")
    print("=" * 80)
    
    try:
        from transformers import pipeline
        
        print("Loading model: tabularisai/eu-pii-safeguard...")
        start_load = time.time()
        ner = pipeline(
            "ner",
            model="tabularisai/eu-pii-safeguard",
            tokenizer="tabularisai/eu-pii-safeguard",
            aggregation_strategy="simple",
            device="cpu",  # Force CPU
        )
        print(f"Model loaded in {time.time() - start_load:.1f}s")
        
        total_detections = 0
        total_time = 0
        
        for i, sample in enumerate(ALL_SAMPLES[:5]):  # Test first 5 for speed
            start_time = time.time()
            results = ner(sample)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Original: {sample[:60]}...")
            print(f"Time: {elapsed*1000:.1f}ms")
            
            if results:
                total_detections += len(results)
                print(f"Detected ({len(results)}):")
                for r in results:
                    print(f"  - {r.get('entity_group', '?')}: {r.get('word', '')} [{r.get('score', 0):.2f}]")
            else:
                print("  No entities detected")
        
        print(f"\n=== Stage 2 Summary ===")
        print(f"Total samples: 5")
        print(f"Total detections: {total_detections}")
        print(f"Total time: {total_time*1000:.1f}ms")
        print(f"Avg per sample: {total_time/5*1000:.1f}ms")
        
    except ImportError as e:
        print(f"SKIPPED: transformers not installed: {e}")
    except Exception as e:
        print(f"ERROR loading model: {e}")


# =============================================================================
# STAGE 3: RUSSIAN NER (Gherman)
# =============================================================================

def test_stage3_gherman():
    """Test Stage 3: Gherman Russian NER model."""
    print("\n" + "=" * 80)
    print("STAGE 3: GHERMAN RUSSIAN NER")
    print("=" * 80)
    
    try:
        from transformers import pipeline
        
        print("Loading model: Gherman/bert-base-NER-Russian...")
        start_load = time.time()
        ner = pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            tokenizer="Gherman/bert-base-NER-Russian",
            aggregation_strategy="simple",
            device="cpu",  # Force CPU
        )
        print(f"Model loaded in {time.time() - start_load:.1f}s")
        
        total_detections = 0
        total_time = 0
        
        for i, sample in enumerate(ALL_SAMPLES[:5]):  # Test first 5 for speed
            start_time = time.time()
            results = ner(sample)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Original: {sample[:60]}...")
            print(f"Time: {elapsed*1000:.1f}ms")
            
            if results:
                total_detections += len(results)
                print(f"Detected ({len(results)}):")
                for r in results:
                    print(f"  - {r.get('entity_group', '?')}: {r.get('word', '')} [{r.get('score', 0):.2f}]")
            else:
                print("  No entities detected")
        
        print(f"\n=== Stage 3 Summary ===")
        print(f"Total samples: 5")
        print(f"Total detections: {total_detections}")
        print(f"Total time: {total_time*1000:.1f}ms")
        print(f"Avg per sample: {total_time/5*1000:.1f}ms")
        
    except ImportError as e:
        print(f"SKIPPED: transformers not installed: {e}")
    except Exception as e:
        print(f"ERROR loading model: {e}")


# =============================================================================
# CASCADE TEST: All 3 stages combined
# Order: Regex → Gherman → EU-PII (with proper overlap resolution)
# =============================================================================

def test_cascade():
    """Test complete 3-stage cascade pipeline with proper overlap resolution."""
    print("\n" + "=" * 80)
    print("CASCADE: REGEX → GHERMAN → EU-PII")
    print("=" * 80)
    
    # Load models
    print("Loading models...")
    
    try:
        from transformers import pipeline
        gherman_ner = pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            tokenizer="Gherman/bert-base-NER-Russian",
            aggregation_strategy="simple",
            device="cpu",
        )
        print("  - Gherman loaded")
    except Exception as e:
        print(f"  - Gherman FAILED: {e}")
        gherman_ner = None
    
    try:
        eu_pii_ner = pipeline(
            "ner",
            model="tabularisai/eu-pii-safeguard",
            tokenizer="tabularisai/eu-pii-safeguard",
            aggregation_strategy="simple",
            device="cpu",
        )
        print("  - EU-PII-Safeguard loaded")
    except Exception as e:
        print(f"  - EU-PII-Safeguard FAILED: {e}")
        eu_pii_ner = None
    
    # Load config from YAML
    config_path = Path(__file__).parent.parent / "config" / "anonymization.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        enabled_entities = set(config.get("entities", []))
        entity_mappings = config.get("entity_mappings", {})
        print(f"Loaded config from: {config_path}")
        print(f"Enabled entities ({len(enabled_entities)}): {', '.join(sorted(enabled_entities))}")
    else:
        print(f"WARNING: Config not found at {config_path}, using defaults")
        enabled_entities = {"PHONE", "EMAIL", "PERSON", "ADDRESS", "CITY", "ORGANIZATION"}
        entity_mappings = {}
    
    # Build unified entity mapping (model_type → semantic_name, model_prefix)
    ENTITY_MAPPING = {}
    
    # Stage 1: Regex
    if "regex" in entity_mappings:
        for model_type, unified in entity_mappings["regex"].items():
            if unified in enabled_entities:
                ENTITY_MAPPING[model_type] = (unified, "REGEX")
    
    # Stage 2: Gherman
    if "gherman" in entity_mappings:
        for model_type, unified in entity_mappings["gherman"].items():
            if unified in enabled_entities:
                ENTITY_MAPPING[model_type] = (unified, "GH")
    
    # Stage 3: EU-PII
    if "eu_pii" in entity_mappings:
        for model_type, unified in entity_mappings["eu_pii"].items():
            if unified in enabled_entities:
                ENTITY_MAPPING[model_type] = (unified, "EU")
    
    def merge_adjacent_entities(entities: list[dict], text: str) -> list[dict]:
        """
        Merge adjacent same-semantic-type entities.
        E.g., "Санкт" + "-" + "Петербург" (all City) → "Санкт-Петербург"
        """
        if not entities:
            return []
        
        merged = []
        for e in entities:
            e['text'] = text[e['start']:e['end']]
        
        entities.sort(key=lambda x: (x['start'], x['end']))
        
        for e in entities:
            if not merged:
                merged.append(e.copy())
                continue
            
            last = merged[-1]
            
            same_type = last.get('semantic') == e.get('semantic')
            adjacent = (last['end'] == e['start']) or (last['end'] == e['start'] + 1 and text[last['end']:e['start']] in (' ', '-', ',', '.'))
            
            if same_type and adjacent:
                last['end'] = max(last['end'], e['end'])
                last['text'] = text[last['start']:last['end']]
            else:
                merged.append(e.copy())
        
        return merged
    
    def resolve_overlaps(entities: list[dict], text_len: int) -> list[dict]:
        """
        Resolve overlaps using the notebook's approach:
        1. Sort by length (longest first), then by start index (earliest)
        2. Use covered_indices to track what's already replaced
        3. Only keep non-overlapping entities
        4. Merge adjacent same-type entities (fragmentation control)
        """
        if not entities:
            return []
        
        # Step 1: Sort by length (longest first), then by start index
        sorted_entities = sorted(entities, key=lambda x: (x['end'] - x['start'], -x['start']), reverse=True)
        
        covered = [False] * text_len
        
        final = []
        for e in sorted_entities:
            start, end = e['start'], e['end']
            
            # Check if overlaps with already covered
            is_overlapping = False
            for idx in range(start, min(end, text_len)):
                if covered[idx]:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                final.append(e)
                # Mark as covered
                for idx in range(start, min(end, text_len)):
                    covered[idx] = True
        
        # Step 2: Merge adjacent same-type entities (fragmentation control)
        # Sort by position first
        final.sort(key=lambda x: x['start'])
        
        merged = []
        for e in final:
            if not merged:
                merged.append(e)
                continue
            
            prev = merged[-1]
            # Check if adjacent (prev ends at or before current starts) AND same semantic type
            # Allow small gap of 1-2 chars for punctuation/spaces
            is_adjacent = prev['end'] >= e['start'] - 2 and prev['semantic'] == e['semantic']
            
            if is_adjacent:
                # Merge: extend previous entity
                prev['text'] = prev['text'] + ' ' + e['text']
                prev['end'] = e['end']
                # Keep the original (regex) if available, prefer regex over NER
                if prev.get('model_prefix') == 'REGEX':
                    pass  # Keep regex
                elif e.get('model_prefix') == 'REGEX':
                    # Replace with regex version
                    prev['text'] = e['text']
                    prev['model_prefix'] = 'REGEX'
            else:
                merged.append(e)
        
        return merged
    
    # Warmup
    print("\nWarming up models...")
    warmup_sample = ALL_SAMPLES[0]
    if gherman_ner:
        _ = gherman_ner(warmup_sample)
    if eu_pii_ner:
        _ = eu_pii_ner(warmup_sample)
    print("Warmup complete.")
    
    num_samples = len(ALL_SAMPLES)
    timers = {'regex': 0.0, 'gherman': 0.0, 'eu_pii': 0.0, 'merge': 0.0, 'total': 0.0}
    total_detections = 0
    total_raw = 0
    
    for i, sample in enumerate(ALL_SAMPLES[:num_samples]):
        all_entities = []
        sample_text = sample
        
        # Stage 1: Regex
        t0 = time.time()
        stage1_entities = RegexPatterns.detect_all(sample_text)
        timers['regex'] += time.time() - t0
        
        for e in stage1_entities:
            if e['type'] in ENTITY_MAPPING:
                mapping = ENTITY_MAPPING[e['type']]
                e['semantic'] = mapping[0]
                e['model_prefix'] = mapping[1]
                all_entities.append(e)
        
        # Stage 2: Gherman
        if gherman_ner:
            t0 = time.time()
            gherman_results = gherman_ner(sample_text)
            timers['gherman'] += time.time() - t0
            
            for r in gherman_results:
                entity_type = r.get('entity_group', '')
                if entity_type and entity_type != 'O' and entity_type in ENTITY_MAPPING:
                    mapping = ENTITY_MAPPING[entity_type]
                    all_entities.append({
                        'type': entity_type,
                        'semantic': mapping[0],
                        'model_prefix': mapping[1],
                        'start': r.get('start', 0),
                        'end': r.get('end', 0),
                        'text': sample_text[r.get('start', 0):r.get('end', 0)],
                        'confidence': r.get('score', 1.0),
                    })
        
        # Stage 3: EU-PII
        if eu_pii_ner:
            t0 = time.time()
            eu_results = eu_pii_ner(sample_text)
            timers['eu_pii'] += time.time() - t0
            
            for r in eu_results:
                entity_type = r.get('entity_group', '')
                if entity_type and entity_type != 'O' and entity_type in ENTITY_MAPPING:
                    mapping = ENTITY_MAPPING[entity_type]
                    all_entities.append({
                        'type': entity_type,
                        'semantic': mapping[0],
                        'model_prefix': mapping[1],
                        'start': r.get('start', 0),
                        'end': r.get('end', 0),
                        'text': sample_text[r.get('start', 0):r.get('end', 0)],
                        'confidence': r.get('score', 1.0),
                    })
        
        total_raw += len(all_entities)
        
        # Resolve overlaps & Merge adjacent
        t0 = time.time()
        unique_entities = resolve_overlaps(all_entities, len(sample_text))
        timers['merge'] += time.time() - t0
        
        # Generate placeholders
        semantic_counters: dict[str, int] = {}
        
        def get_placeholder(semantic_type: str) -> str:
            if semantic_type not in semantic_counters:
                semantic_counters[semantic_type] = 0
            idx = semantic_counters[semantic_type]
            semantic_counters[semantic_type] += 1
            if idx < 26:
                letter = chr(ord('A') + idx)
            else:
                letter = chr(ord('A') + (idx // 26) - 1) + chr(ord('A') + (idx % 26))
            return f"[{semantic_type} {letter}]"
        
        for e in unique_entities:
            e['placeholder'] = get_placeholder(e['semantic'])
        
        # Apply replacements
        anonymized = sample_text
        for e in reversed(unique_entities):
            anonymized = anonymized[:e['start']] + e['placeholder'] + anonymized[e['end']:]
        
        total_detections += len(unique_entities)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Original: {sample_text[:80]}...")
        print(f"Anonymized: {anonymized[:100]}...")
        print(f"Entities: {len(all_entities)} raw → {len(unique_entities)} final")
        for e in unique_entities:
            print(f"  - {e['semantic']} ({e['model_prefix']}): {e['text'][:30]} → {e['placeholder']}")
    
    print(f"\n{'='*60}")
    print("CASCADE TIMING BREAKDOWN")
    print(f"{'='*60}")
    print(f"Stage 1 (Regex):    {timers['regex']*1000:.1f}ms total, {timers['regex']/num_samples*1000:.2f}ms/sample")
    print(f"Stage 2 (Gherman): {timers['gherman']*1000:.1f}ms total, {timers['gherman']/num_samples*1000:.1f}ms/sample")
    print(f"Stage 3 (EU-PII):  {timers['eu_pii']*1000:.1f}ms total, {timers['eu_pii']/num_samples*1000:.1f}ms/sample")
    print(f"Merge + Dedup:      {timers['merge']*1000:.1f}ms total, {timers['merge']/num_samples*1000:.2f}ms/sample")
    print(f"{'='*60}")
    total_cascade = timers['regex'] + timers['gherman'] + timers['eu_pii'] + timers['merge']
    print(f"CASCADE TOTAL:      {total_cascade*1000:.1f}ms total, {total_cascade/num_samples*1000:.1f}ms/sample")
    print(f"{'='*60}")
    print(f"Raw detections:    {total_raw}")
    print(f"Final detections:  {total_detections}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ANONYMIZATION PIPELINE TEST SCRIPT")
    print("=" * 80)
    print(f"Testing {len(ALL_SAMPLES)} samples")
    
    # Stage 1: Always works (regex only)
    test_stage1_regex()
    
    # Stage 2: Requires transformers + model
    test_stage2_eu_pii()
    
    # Stage 3: Requires transformers + model
    test_stage3_gherman()
    
    # Cascade: All stages combined
    test_cascade()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
