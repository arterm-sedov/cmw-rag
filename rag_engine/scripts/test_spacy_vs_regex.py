#!/usr/bin/env python3
"""
Test: Presidio with Russian spaCy models vs Just Regex
Comparing Presidio's value with spaCy NER vs plain regex.

Tests:
1. Presidio (ru_core_news_lg + PatternRecognizers) 
2. Just Regex (like current cascade Stage 1)
"""

import time
import re
from pathlib import Path

# Test samples
RUSSIAN_SAMPLES = [
    "Иван Иванов, +7-900-123-45-67, ivan.ivanov@example.com, Москва, ул. Ленина, д. 10, кв. 5. Работает менеджером в ООО 'Рога и Копыта'.",
    "Анна Смирнова, +7-901-234-56-78, anna.smirnova@company.ru, Санкт-Петербург, Невский пр-т, д. 25, оф. 10. Является директором в 'ТехноПрогресс'.",
    "Дмитрий Петров, +7-902-345-67-89, dmitry.petrov@mail.org, Новосибирск, ул. Кирова, д. 3, кв. 1. Инженер-программист в АО 'СибСофт'.",
]

EDGE_CASES = [
    "Паспорт: 45 77 123456, ИНН: 1234567890, СНИЛС: 123-456-789 00",
    "ОГРН: 1234567890123, КПП: 123456789, ОКПО: 12345678, ОКВЭД: 62.01",
    "БИК: 044525700, р/с: 40702810380000001234, к/с: 30101810400000000700",
    "Госномер: А777АА77, VIN: WA1LAAFPNKD123456",
    "Пароль: SuperSecret123, логин: admin, API ключ: sk_live_abc123xyz",
    "Мне 35 лет, родился 15.03.1990.",
]

ALL_SAMPLES = RUSSIAN_SAMPLES + EDGE_CASES


def test_presidio_with_spacy():
    """Test Presidio with ru_core_news_lg + Custom PatternRecognizers."""
    print("\n" + "=" * 80)
    print("PRESIDIO + ru_core_news_lg (Russian NER) + PatternRecognizers")
    print("=" * 80)
    
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as e:
        print(f"SKIPPED: Presidio not installed: {e}")
        return
    
    # Download ru_core_news_lg if needed
    import spacy
    try:
        nlp = spacy.load("ru_core_news_lg")
        print("  - ru_core_news_lg loaded")
    except OSError:
        print("  - Downloading ru_core_news_lg (large model, ~80MB)...")
        import spacy.cli
        spacy.cli.download("ru_core_news_lg")
        nlp = spacy.load("ru_core_news_lg")
    
    # Custom PatternRecognizers (same as before)
    class RussianPhoneRecognizer(PatternRecognizer):
        def __init__(self):
            patterns = [
                Pattern(name="phone_1", regex=r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', score=0.9),
                Pattern(name="phone_2", regex=r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', score=0.9),
                Pattern(name="phone_3", regex=r'\+7\d{10}', score=0.85),
                Pattern(name="phone_4", regex=r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', score=0.9),
                Pattern(name="phone_5", regex=r'8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', score=0.9),
                Pattern(name="phone_6", regex=r'8\d{10}', score=0.85),
            ]
            super().__init__(
                supported_entity="PHONE_RU",
                patterns=patterns,
                context=["телефон", "тел", "мобильный"],
                supported_language="ru",
            )

    class RussianPassportRecognizer(PatternRecognizer):
        def __init__(self):
            patterns = [
                Pattern(name="passport_1", regex=r'\b\d{2}\s+\d{2}\s+\d{6}\b', score=0.85),
                Pattern(name="passport_2", regex=r'\b\d{4}\s+\d{6}\b', score=0.85),
            ]
            super().__init__(
                supported_entity="PASSPORT_RU",
                patterns=patterns,
                context=["паспорт"],
                supported_language="ru",
            )

    class RussianSnilsRecognizer(PatternRecognizer):
        def __init__(self):
            patterns = [Pattern(name="snils", regex=r'\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b', score=0.9)]
            super().__init__(supported_entity="SNILS_RU", patterns=patterns, context=["снилс", "страховой"], supported_language="ru")

    class RussianInnRecognizer(PatternRecognizer):
        def __init__(self):
            patterns = [Pattern(name="inn", regex=r'\b\d{10}\b', score=0.8), Pattern(name="inn_12", regex=r'\b\d{12}\b', score=0.8)]
            super().__init__(supported_entity="INN_RU", patterns=patterns, context=["инн", "налог"], supported_language="ru")

    class RussianCarNumberRecognizer(PatternRecognizer):
        def __init__(self):
            patterns = [Pattern(name="car_number", regex=r'\b[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}\b', score=0.85)]
            super().__init__(supported_entity="CAR_NUMBER_RU", patterns=patterns, context=["номер", "госномер"], supported_language="ru")

    # Build registry
    registry = RecognizerRegistry(supported_languages=["ru"])
    
    # Add PatternRecognizers
    registry.add_recognizer(RussianPhoneRecognizer())
    registry.add_recognizer(RussianPassportRecognizer())
    registry.add_recognizer(RussianSnilsRecognizer())
    registry.add_recognizer(RussianInnRecognizer())
    registry.add_recognizer(RussianCarNumberRecognizer())
    print("  - Custom PatternRecognizers ✓")
    
    # Add built-in email
    from presidio_analyzer.predefined_recognizers import EmailRecognizer
    registry.add_recognizer(EmailRecognizer())
    print("  - EmailRecognizer ✓")
    
    # Setup NLP engine with ru_core_news_lg
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ru", "model_name": "ru_core_news_lg"}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
    print("  - spaCy NLP Engine: ru_core_news_lg ✓")
    
    # Create analyzer
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["ru"],
    )
    
    print("\nRunning Presidio + spaCy on samples...")
    
    total_time = 0
    total_detections = 0
    
    for i, sample in enumerate(ALL_SAMPLES):
        start_time = time.time()
        results = analyzer.analyze(text=sample, language="ru", score_threshold=0.5)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample[:50]}...")
        print(f"Time: {elapsed*1000:.1f}ms, Entities: {len(results)}")
        
        if results:
            total_detections += len(results)
            for r in results:
                entity_text = sample[r.start:r.end] if r.start < len(sample) else ""
                print(f"  - {r.entity_type}: '{entity_text}' [{r.score:.2f}]")
    
    print(f"\n{'='*60}")
    print("PRESIDIO + SPACY TIMING")
    print(f"{'='*60}")
    print(f"Total samples: {len(ALL_SAMPLES)}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {total_time*1000:.1f}ms")
    print(f"Avg per sample: {total_time/len(ALL_SAMPLES)*1000:.1f}ms")
    print(f"{'='*60}")


def test_regex_only():
    """Test just regex patterns (like current cascade Stage 1)."""
    print("\n" + "=" * 80)
    print("REGEX ONLY (Current Cascade Stage 1)")
    print("=" * 80)
    
    # Same regex patterns as in test_anonymization_stages.py
    PHONE_PATTERNS = [
        re.compile(r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'\+7\d{10}'),
        re.compile(r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'),
        re.compile(r'8\d{10}'),
    ]
    
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    PASSPORT_PATTERN = re.compile(r'\b\d{2}\s+\d{2}\s+\d{6}\b|\b\d{4}\s+\d{6}\b')
    SNILS_PATTERN = re.compile(r'\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b')
    INN_PATTERN = re.compile(r'\b\d{10}\b|\b\d{12}\b')
    OGRN_PATTERN = re.compile(r'\b\d{13}\b|\b\d{15}\b')
    KPP_PATTERN = re.compile(r'\b\d{9}\b')
    OKPO_PATTERN = re.compile(r'\b\d{8}\b|\b\d{10}\b')
    OKVED_PATTERN = re.compile(r'\b\d{2}\.\d{2}(\.\d{1,2})?\b')
    BIC_PATTERN = re.compile(r'\b0[4-5]\d{7}\b')
    BANK_ACCOUNT_PATTERN = re.compile(r'\b40[5-8]\d{17}\b|\b301\d{17}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
    CAR_NUMBER_PATTERN = re.compile(r'\b[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}\b')
    VIN_PATTERN = re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b')
    PASSWORD_PATTERN = re.compile(r'(пароль|password|passwd|pwd)[_\s:=]+[^\s]+', re.IGNORECASE)
    LOGIN_PATTERN = re.compile(r'(логин|login|username)[_\s:=]+[^\s]+', re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|apikey|ключ)[_\s:=]+[^\s]+', re.IGNORECASE)
    URL_PATTERN = re.compile(r'https?://[^\s/$.?#].[^\s,]*', re.IGNORECASE)
    IP_ADDRESS_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    AGE_PATTERN = re.compile(r'мне\s+\d{1,2}\s+лет', re.IGNORECASE)
    
    def detect_all(text):
        detections = []
        
        def add_if_no_overlap(pattern, type_name, context_keywords=None):
            for match in pattern.finditer(text):
                overlapping = any(
                    not (match.end() <= d['start'] or match.start() >= d['end'])
                    for d in detections
                )
                if overlapping:
                    continue
                if context_keywords:
                    ctx_start = max(0, match.start() - 20)
                    context = text[ctx_start:match.end()].lower()
                    if not any(kw in context for kw in context_keywords):
                        continue
                detections.append({
                    "type": type_name, "text": match.group(),
                    "start": match.start(), "end": match.end(),
                })
        
        # Priority order
        add_if_no_overlap(BANK_ACCOUNT_PATTERN, "BANK_ACCOUNT")
        add_if_no_overlap(VIN_PATTERN, "VIN")
        add_if_no_overlap(EMAIL_PATTERN, "EMAIL")
        add_if_no_overlap(PASSWORD_PATTERN, "PASSWORD")
        add_if_no_overlap(LOGIN_PATTERN, "LOGIN")
        add_if_no_overlap(API_KEY_PATTERN, "API_KEY")
        
        for pattern in PHONE_PATTERNS:
            add_if_no_overlap(pattern, "PHONE")
        add_if_no_overlap(CREDIT_CARD_PATTERN, "CREDIT_CARD")
        
        add_if_no_overlap(SNILS_PATTERN, "SNILS")
        add_if_no_overlap(PASSPORT_PATTERN, "PASSPORT")
        add_if_no_overlap(CAR_NUMBER_PATTERN, "CAR_NUMBER")
        
        add_if_no_overlap(INN_PATTERN, "INN", ['инн', 'inn', 'налог'])
        add_if_no_overlap(OGRN_PATTERN, "OGRN", ['огрн', 'ogrn'])
        add_if_no_overlap(KPP_PATTERN, "KPP", ['кпп', 'kpp'])
        add_if_no_overlap(BIC_PATTERN, "BIC", ['бик', 'bic'])
        
        add_if_no_overlap(URL_PATTERN, "URL")
        add_if_no_overlap(IP_ADDRESS_PATTERN, "IP_ADDRESS")
        add_if_no_overlap(AGE_PATTERN, "AGE")
        
        detections.sort(key=lambda x: x['start'])
        return detections
    
    total_time = 0
    total_detections = 0
    
    for i, sample in enumerate(ALL_SAMPLES):
        start_time = time.time()
        detections = detect_all(sample)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample[:50]}...")
        print(f"Time: {elapsed*1000:.1f}ms, Entities: {len(detections)}")
        
        if detections:
            total_detections += len(detections)
            for d in detections:
                print(f"  - {d['type']}: '{d['text']}'")
    
    print(f"\n{'='*60}")
    print("REGEX ONLY TIMING")
    print(f"{'='*60}")
    print(f"Total samples: {len(ALL_SAMPLES)}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {total_time*1000:.1f}ms")
    print(f"Avg per sample: {total_time/len(ALL_SAMPLES)*1000:.1f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 80)
    print("COMPARISON: Presidio+spaCy vs Regex Only")
    print("=" * 80)
    
    # Test 1: Presidio + spaCy
    test_presidio_with_spacy()
    
    # Test 2: Regex only
    test_regex_only()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
