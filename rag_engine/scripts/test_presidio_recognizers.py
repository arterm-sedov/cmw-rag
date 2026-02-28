#!/usr/bin/env python3
"""
Test: Presidio-based approach with TransformersRecognizers
Comparing with our custom cascade pipeline.

Architecture:
├── Presidio Analyzer Engine
│   ├── Stage 1: Custom PatternRecognizers (Russian PII)
│   ├── Stage 2: TransformersRecognizer(Gherman)
│   └── Stage 3: TransformersRecognizer(EU-PII-Safeguard)
└── Built-in merge/dedupe from Presidio
"""

import time
import re
from pathlib import Path

# Test samples (from original testbed)
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
    "Мне 35 лет, родился 15.03.1990.",
]

ALL_SAMPLES = RUSSIAN_SAMPLES + EDGE_CASES


def test_presidio_recognizers():
    """Test Presidio with transformers NLP engine for NER."""
    print("\n" + "=" * 80)
    print("PRESIDIO: TransformersNlpEngine (Gherman) + Custom PatternRecognizers")
    print("=" * 80)
    
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        import spacy
    except ImportError as e:
        print(f"SKIPPED: Presidio not installed: {e}")
        return
    
    # Download spacy model if needed
    try:
        spacy.load("ru_core_news_sm")
    except OSError:
        print("Downloading ru_core_news_sm...")
        import spacy.cli
        spacy.cli.download("ru_core_news_sm")
    
    # =============================================================================
    # Stage 1: Custom PatternRecognizers for Russian PII
    # =============================================================================
    
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

    # =============================================================================
    # Stage 2 & 3: TransformersRecognizers
    # =============================================================================
    
    # Lazy loading for transformers recognizers
    gherman_recognizer = None
    eu_pii_recognizer = None
    
    def get_gherman_recognizer():
        nonlocal gherman_recognizer
        if gherman_recognizer is None:
            try:
                from presidio_analyzer import TransformersRecognizer
                
                # Entity mapping for Gherman
                gherman_config = {
                    "model_path": "Gherman/bert-base-NER-Russian",
                    "aggregation_strategy": "simple",
                    "device": "cpu",
                    "labels_to_ignore": ["O"],
                    "model_to_presidio_entity_mapping": {
                        "PER": "PERSON",
                        "LOC": "LOCATION",
                        "ORG": "ORGANIZATION",
                    },
                }
                
                gherman_recognizer = TransformersRecognizer(
                    model_path="Gherman/bert-base-NER-Russian",
                    supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],
                )
                gherman_recognizer.load_transformer(**gherman_config)
                print("  - Gherman/bert-base-NER-Russian loaded")
            except Exception as e:
                print(f"  - Gherman FAILED: {e}")
                return None
        return gherman_recognizer
    
    def get_eu_pii_recognizer():
        nonlocal eu_pii_recognizer
        if eu_pii_recognizer is None:
            try:
                from presidio_analyzer import TransformersRecognizer
                
                eu_pii_recognizer = TransformersRecognizer(
                    model_path="tabularisai/eu-pii-safeguard",
                    supported_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "ORGANIZATION"],
                )
                eu_pii_recognizer.load_transformer(
                    model_path="tabularisai/eu-pii-safeguard",
                    aggregation_strategy="simple",
                    device="cpu",
                )
                print("  - EU-PII-Safeguard loaded")
            except Exception as e:
                print(f"  - EU-PII-Safeguard FAILED: {e}")
                return None
        return eu_pii_recognizer
    
    # =============================================================================
    # Build Presidio Analyzer with transformers NLP engine (Gherman for Russian NER)
    # =============================================================================
    
    print("\nSetting up Presidio Analyzer Engine...")
    
    # Create registry with supported languages
    registry = RecognizerRegistry(supported_languages=["ru"])
    
    # Add Stage 1: Custom PatternRecognizers (for Russian PII) - These work!
    registry.add_recognizer(RussianPhoneRecognizer())
    registry.add_recognizer(RussianPassportRecognizer())
    registry.add_recognizer(RussianSnilsRecognizer())
    registry.add_recognizer(RussianInnRecognizer())
    registry.add_recognizer(RussianCarNumberRecognizer())
    print("  - Stage 1: Custom PatternRecognizers (Russian PII) ✓")
    
    # Add built-in email recognizer
    from presidio_analyzer.predefined_recognizers import EmailRecognizer
    registry.add_recognizer(EmailRecognizer())
    print("  - Built-in: EmailRecognizer ✓")
    
    # Setup transformers NLP engine with Gherman model
    try:
        transformers_config = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "ru",
                    "model_name": {
                        "spacy": "ru_core_news_sm",  # For tokenization, lemmas
                        "transformers": "Gherman/bert-base-NER-Russian",  # For NER
                    },
                }
            ],
            "ner_model_configuration": {
                "labels_to_ignore": ["O"],
                "model_to_presidio_entity_mapping": {
                    "PER": "PERSON",
                    "LOC": "LOCATION", 
                    "ORG": "ORGANIZATION",
                },
                "aggregation_strategy": "simple",
            },
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=transformers_config).create_engine()
        print("  - Transformers NLP Engine: Gherman/bert-base-NER-Russian ✓")
    except Exception as e:
        print(f"  - Transformers NLP Engine FAILED: {e}")
        print("  - Falling back to spaCy only...")
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "ru", "model_name": "ru_core_news_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        print("  - spaCy NLP Engine: ru_core_news_sm (fallback)")
    
    # Create analyzer
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["ru"],
    )
    
    print("\nRunning Presidio Analyzer on samples...")
    
    total_time = 0
    total_detections = 0
    
    for i, sample in enumerate(ALL_SAMPLES):
        start_time = time.time()
        
        # Run analysis - Presidio handles cascade, deduplication internally
        results = analyzer.analyze(text=sample, language="ru", score_threshold=0.5)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample[:60]}...")
        print(f"Time: {elapsed*1000:.1f}ms")
        print(f"Entities found: {len(results)}")
        
        if results:
            total_detections += len(results)
            for r in results:
                entity_text = sample[r.start:r.end] if r.start < len(sample) else ""
                print(f"  - {r.entity_type}: '{entity_text}' [{r.score:.2f}]")
    
    print(f"\n{'='*60}")
    print("PRESIDIO TIMING")
    print(f"{'='*60}")
    print(f"Total samples: {len(ALL_SAMPLES)}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {total_time*1000:.1f}ms")
    print(f"Avg per sample: {total_time/len(ALL_SAMPLES)*1000:.1f}ms")
    print(f"{'='*60}")


def test_current_cascade():
    """Run the current custom cascade pipeline for comparison."""
    print("\n" + "=" * 80)
    print("CURRENT CASCADE: Regex → Gherman → EU-PII (Custom Logic)")
    print("=" * 80)
    
    # Import the testbed functions
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Replicate the cascade test logic
    from test_anonymization_stages import RegexPatterns, ALL_SAMPLES
    
    try:
        from transformers import pipeline
        
        print("Loading models...")
        
        gherman_ner = pipeline(
            "ner",
            model="Gherman/bert-base-NER-Russian",
            tokenizer="Gherman/bert-base-NER-Russian",
            aggregation_strategy="simple",
            device="cpu",
        )
        print("  - Gherman loaded")
        
        eu_pii_ner = pipeline(
            "ner",
            model="tabularisai/eu-pii-safeguard",
            tokenizer="tabularisai/eu-pii-safeguard",
            aggregation_strategy="simple",
            device="cpu",
        )
        print("  - EU-PII-Safeguard loaded")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Simple entity mapping
    ENTITY_MAPPING = {
        "PER": ("PERSON", "GH"),
        "LOC": ("LOCATION", "GH"),
        "ORG": ("ORGANIZATION", "GH"),
        "EMAIL_ADDRESS": ("EMAIL", "EU"),
        "PHONE_NUMBER": ("PHONE", "EU"),
        "PERSON": ("PERSON", "EU"),
    }
    
    def resolve_overlaps(entities, text_len):
        """Same as original testbed."""
        if not entities:
            return []
        
        sorted_entities = sorted(entities, key=lambda x: (x['end'] - x['start'], -x['start']), reverse=True)
        covered = [False] * text_len
        
        final = []
        for e in sorted_entities:
            start, end = e['start'], e['end']
            is_overlapping = any(covered[idx] for idx in range(start, min(end, text_len)))
            
            if not is_overlapping:
                final.append(e)
                for idx in range(start, min(end, text_len)):
                    covered[idx] = True
        
        return sorted(final, key=lambda x: x['start'])
    
    total_time = 0
    total_detections = 0
    
    for i, sample in enumerate(ALL_SAMPLES):
        all_entities = []
        
        # Stage 1: Regex
        t0 = time.time()
        stage1 = RegexPatterns.detect_all(sample)
        
        for e in stage1:
            if e['type'] in {"PHONE", "EMAIL", "PASSPORT", "SNILS", "INN", "CAR_NUMBER"}:
                e['semantic'] = e['type']
                e['model_prefix'] = 'REGEX'
                all_entities.append(e)
        
        # Stage 2: Gherman
        try:
            gherman_results = gherman_ner(sample)
            for r in gherman_results:
                entity_type = r.get('entity_group', '')
                if entity_type != 'O' and entity_type in ENTITY_MAPPING:
                    mapping = ENTITY_MAPPING[entity_type]
                    all_entities.append({
                        'semantic': mapping[0],
                        'model_prefix': mapping[1],
                        'start': r.get('start', 0),
                        'end': r.get('end', 0),
                    })
        except Exception as e:
            print(f"  Gherman error: {e}")
        
        # Stage 3: EU-PII
        try:
            eu_results = eu_pii_ner(sample)
            for r in eu_results:
                entity_type = r.get('entity_group', '')
                if entity_type != 'O' and entity_type in ENTITY_MAPPING:
                    mapping = ENTITY_MAPPING[entity_type]
                    all_entities.append({
                        'semantic': mapping[0],
                        'model_prefix': mapping[1],
                        'start': r.get('start', 0),
                        'end': r.get('end', 0),
                    })
        except Exception as e:
            print(f"  EU-PII error: {e}")
        
        # Resolve overlaps
        unique = resolve_overlaps(all_entities, len(sample))
        
        elapsed = time.time() - t0
        total_time += elapsed
        total_detections += len(unique)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample[:60]}...")
        print(f"Time: {elapsed*1000:.1f}ms, Entities: {len(unique)}")
        
        for e in unique[:5]:
            entity_text = sample[e['start']:e['end']] if e['start'] < len(sample) else ""
            print(f"  - {e['semantic']} ({e['model_prefix']}): '{entity_text}'")
    
    print(f"\n{'='*60}")
    print("CURRENT CASCADE TIMING")
    print(f"{'='*60}")
    print(f"Total samples: {len(ALL_SAMPLES)}")
    print(f"Total detections: {total_detections}")
    print(f"Total time: {total_time*1000:.1f}ms")
    print(f"Avg per sample: {total_time/len(ALL_SAMPLES)*1000:.1f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 80)
    print("COMPARISON: Presidio vs Current Cascade")
    print("=" * 80)
    
    # Test Presidio approach
    test_presidio_recognizers()
    
    # Test current cascade approach  
    test_current_cascade()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
