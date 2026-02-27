# Anonymization Pipeline Implementation Plan

**Date:** 2026-02-27  
**Version:** 1.0  
**Status:** Draft for Review  
**Author:** OpenCode Agent  
**Based on:** User requirements + rus-anonymizer + Microsoft Presidio + tabularisai/eu-pii-safeguard + Gherman/bert-base-NER-Russian research

---

## 1. Executive Summary

Implement a **cascaded reversible anonymization pipeline** that:

1. **Anonymizes** user messages before LLM processing (before guardian check)
2. **Deanonymizes** LLM responses before UI display
3. Provides **transparent user experience** - users see original text, LLM sees anonymized
4. Is **opt-in** via environment configuration
5. Supports both **regular chat UI** and **Comindware platform integration**

### Architecture Overview

```
User Message
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Deterministic/Regex (Presidio + Custom Russian) │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Presidio Built-in: EMAIL, CREDIT_CARD, URL, IP     │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Custom Russian Recognizers:                          │  │
│  │ - Phone RU (+7, 8, multiple formats)                │  │
│  │ - Passport RU, SNILS, INN                          │  │
│  │ - Car numbers, Age, Short names                    │  │
│  │ - Password, Login, API keys, Internal URLs         │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: EU NER (tabularisai/eu-pii-safeguard)           │
│  - 42 entity types, 26 European languages                   │
│  - Multilingual NER: PERSON, ORG, LOC (works for Russian)  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Russian NER (Gherman/bert-base-NER-Russian)      │
│  - Russian-specific NER for entities missed by Stage 2      │
│  - Complements EU NER for Russian text                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Deterministic/Regex                               │
│  - Presidio + Russian regexes (phones, emails, passports,   │
│    SNILS, INN, car numbers, corporate credentials, etc.)    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: EU NER (tabularisai/eu-pii-safeguard)            │
│  - 42 entity types, 26 European languages                    │
│  - Multilingual NER: PERSON, ORG, LOC (works for Russian)   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Russian NER (Gherman/bert-base-NER-Russian)       │
│  - Russian-specific NER for entities missed by Stage 2      │
│  - Complements EU NER for Russian text                      │
└─────────────────────────────────────────────────────────────┘
    ↓
[Guardian Check - Content Moderation - existing]
    ↓
[LLM Processing - with anonymized content]
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Deanonymization                                           │
│  - Restore original PII from mapping                       │
│  - Handle LLM output containing placeholders              │
└─────────────────────────────────────────────────────────────┘
    ↓
User sees original text (transparent experience)
```

---

## 2. Component Design

### 2.1 Directory Structure

```
rag_engine/
├── anonymization/
│   ├── __init__.py
│   ├── config.py                 # YAML config loader
│   ├── pipeline.py               # Main orchestrator
│   ├── types.py                  # Data classes, enums
│   ├── mapping_store.py          # Unified PII mapping storage
│   ├── helpers.py                # Unified helper functions
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py              # Base stage interface
│   │   ├── stage1_regex.py       # Presidio + Russian regexes
│   │   ├── stage2_eu_pii.py     # EU-PII-Safeguard
│   │   └── stage3_ner.py        # Russian NER
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── factory.py           # Inference provider factory
│   │   ├── vllm_client.py       # vLLM HTTP client
│   │   ├── mosec_client.py     # MOSEC HTTP client
│   │   └── local.py            # Direct torch inference
│   ├── helpers.py               # Unified helper functions
│   └── prompts.py               # LLM wrapper prompts
├── tests/
│   └── test_anonymization/
│       ├── __init__.py
│       ├── test_pipeline.py
│       ├── test_stages.py
│       ├── test_reversibility.py
│       ├── test_integration.py
│       └── fixtures/
│           ├── samples.py       # Test samples
│           └── datasets/        # jayguard samples
└── scripts/
    ├── test_anonymization.py    # Manual testing script
    └── benchmark_anonymization.py
```

### 2.2 Configuration Files

#### 2.2.1 YAML Config: `rag_engine/config/anonymization.yaml`

```yaml
# =============================================================================
# Anonymization Pipeline Configuration
# =============================================================================
# This file defines the anonymization pipeline behavior.
# Secrets go to .env, structured config goes here.
# =============================================================================

# Pipeline-wide settings
pipeline:
  # Enable/disable entire pipeline
  enabled: false  # Set to true in .env to enable
  
  # Detection aggressiveness level
  # conservative: Only high-confidence detections
  # balanced: Moderate coverage (recommended)
  # aggressive: Maximum coverage, may have false positives
  detection_level: "balanced"
  
  # Whether to run stages in cascade (failover) or parallel
  mode: "cascade"  # cascade | parallel
  
  # Placeholder format for reversible anonymization
  # Options:
  # - numeric: "{entity_type}_{counter}"  e.g., NAME_0, EMAIL_1
  # - semantic: "Person A", "Email B", "Phone 1" (more readable for LLM)
  placeholder_format: "semantic"  # Recommended: semantic is more readable for LLM
  
  # Two-phase anonymization (from DataAnonymiser approach):
  # - Phase 1: Scan text, count unique entities, assign consistent replacements
  # - Phase 2: Replace all entities with assigned replacements
  # Benefits: Same original text → same placeholder throughout (e.g., "Иван" always becomes "Person A")
  two_phase_anonymization: true
  
  # Handle Unicode dash variants (important for Russian text)
  # See: https://github.com/levitation-opensource/DataAnonymiser
  # Handles: hyphen (-), non-breaking hyphen (‑), en-dash (–), em-dash (—), etc.
  handle_unicode_dashes: true

# Stage 1: Deterministic/Regex (Presidio + Custom Russian Recognizers)
# =============================================================================
# Stage 1 uses Microsoft Presidio as the orchestrator, extended with custom
# Russian-specific recognizers from:
# - https://github.com/JohnConnor123/rus-anonymizer
# - https://github.com/ranas-mukminov/ru-smb-pd-anonymizer
#
# Architecture:
# ├── Presidio Built-in Recognizers (English/Latin)
# │   ├── EMAIL_ADDRESS, CREDIT_CARD, URL, IP_ADDRESS, DATE_TIME
# │   └── PHONE (generic international)
# │
# └── Custom Presidio Recognizers (Russian-specific)
#     ├── RussianPhoneRecognizer (+7, 8, various formats)
#     ├── RussianPassportRecognizer (XX XX XXXXXX)
#     ├── RussianSnilsRecognizer (XXX-XXX-XXX XX)
#     ├── RussianInnRecognizer (10/12 digits)
#     ├── RussianCarNumberRecognizer (А777АА777)
#     ├── RussianAgeRecognizer ("мне X лет")
#     └── Context words: телефон, паспорт, снилс, инн, etc.
#
# Benefits:
# - Single framework (Presidio) as orchestrator
# - Battle-tested infrastructure
# - Easy to extend with more custom recognizers
# - Russian-specific patterns from proven open-source sources
# =============================================================================
stage1_regex:
  enabled: true
  
  # =============================================================================
  # SIMPLE ENTITY LIST - just list what to anonymize
  # Empty = all enabled. Comment/uncomment to configure.
  # This is the simplest way to opt-in/out specific PII types
  # =============================================================================
  entities:
    # Contact (usually always needed)
    - EMAIL
    - PHONE
    
    # Personal IDs (Russian)
    - PASSPORT
    - SNILS
    - INN
    
    # Corporate/Financial (Russian)
    - OGRN
    - KPP
    - OKPO
    - OKVED
    - BIC
    - BANK_ACCOUNT
    
    # Vehicles
    - CAR_NUMBER
    - VIN
    
    # Sensitive/Corporate
    - PASSWORD
    - LOGIN
    - API_KEY
    - INTERNAL_URL
    
    # Optional - uncomment if needed
    # - DATE_TIME
    # - BANK_CARD
    # - OMS_POLICY
    # - DRIVER_LICENSE
    # - BIRTH_CERT
    # - MILITARY_ID
    # - OKATO
  
  # Presidio orchestrator settings
  presidio:
    # Entities to detect via Presidio built-in recognizers
    # These handle Latin/English patterns
    # NOTE: DATE_TIME is NOT included by default - dates are not sensitive PII
    # and anonymizing them can hinder LLM context understanding
    built_in_entities:
      - EMAIL_ADDRESS
      - CREDIT_CARD
      - URL
      - IP_ADDRESS
      # - DATE_TIME  # Disabled by default - not sensitive, hurts LLM context
    
    # Context words to enhance detection (language-specific)
    context_words:
      ru:
        email: ["email", "эл. почта", "e-mail", "почта"]
        phone: ["тел", "телефон", "мобильный", "контактный"]
        passport: ["паспорт", "свид-во"]
      en:
        email: ["email", "e-mail"]
        phone: ["phone", "mobile", "contact"]
  
  # Custom Russian Recognizers (extend Presidio)
  # =============================================================================
  # Per-entity enable/disable: set "enabled: false" for any recognizer you want to skip
  # Example: To disable car plate detection, set enabled: false for RussianCarNumberRecognizer
  # This gives fine-grained control over which PII types are anonymized
  # =============================================================================
  custom_recognizers:
    enabled: true
    
    # Russian phone patterns (6 formats from rus-anonymizer)
    - name: "RussianPhoneRecognizer"
      enabled: true  # Set to false to disable
      entity: "PHONE_RU"
      patterns:
        - '\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'  # +7 (XXX) XXX-XX-XX
        - '\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'      # +7 XXX XXX-XX-XX
        - '8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'    # 8 (XXX) XXX-XX-XX
        - '8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'        # 8 XXX XXX-XX-XX
        - '\+7\d{10}'                                        # +7XXXXXXXXXX
        - '8\d{10}'                                          # 8XXXXXXXXXX
      context: ["телефон", "тел", "мобильный", "контактный", "связи"]
      confidence: 0.9
    
    # Russian passport (series + number)
    - name: "RussianPassportRecognizer"
      enabled: true
      entity: "PASSPORT_RU"
      patterns:
        - '\b\d{2}\s+\d{2}\s+\d{6}\b'  # XX XX XXXXXX
        - '\b\d{4}\s+\d{6}\b'           # XXXX XXXXXX
      context: ["паспорт", "паспорта", "паспортом"]
      confidence: 0.85
    
    # Russian SNILS (insurance number)
    - name: "RussianSnilsRecognizer"
      entity: "SNILS_RU"
      patterns:
        - '\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b'  # XXX-XXX-XXX XX
      context: ["снилс", "СНИЛС", "страховой"]
      confidence: 0.9
    
    # Russian INN (tax ID)
    - name: "RussianInnRecognizer"
      entity: "INN_RU"
      patterns:
        - '\b\d{10}\b'  # 10 digits (legal entity)
        - '\b\d{12}\b'  # 12 digits (individual)
      context: ["инн", "ИНН", "налоговый"]
      confidence: 0.8
    
    # Russian OGRN (State Registration Number) - from research
    - name: "RussianOgrnRecognizer"
      entity: "OGRN_RU"
      patterns:
        - '\b\d{13}\b'  # OGRN (legal entity, 13 digits)
        - '\b\d{15}\b'  # OGRNIP (individual entrepreneur, 15 digits)
      context: ["огрн", "ОГРН", "огрнип", "ОГРНИП", "госрегистрация"]
      confidence: 0.85
    
    # Russian KPP (Tax Registration Reason Code) - from research
    - name: "RussianKppRecognizer"
      entity: "KPP_RU"
      patterns:
        - '\b\d{9}\b'  # 9 digits
      context: ["кпп", "КПП", "причина постановки"]
      confidence: 0.8
    
    # Russian national classifiers (ОКПО, ОКВЭД, etc.) - from research
    - name: "RussianOkpoRecognizer"
      entity: "OKPO_RU"
      patterns:
        - '\b\d{8}\b'  # ОКПО legal (8 digits)
        - '\b\d{10}\b'  # ОКПО IE (10 digits)
      context: ["окпо", "ОКПО", "классификатор"]
      confidence: 0.7
    
    - name: "RussianOkvedRecognizer"
      entity: "OKVED_RU"
      patterns:
        - '\b\d{2}\.\d{2}\.\d{2}\b'  # XX.XX.XX
        - '\b\d{1,2}\.\d{2}\.\d{2}\b'  # X.XX.XX
      context: ["оквэд", "ОКВЭД", "вид деятельности"]
      confidence: 0.7
    
    # Russian BIC (Bank Identification Code) - from research
    - name: "RussianBicRecognizer"
      entity: "BIC_RU"
      patterns:
        - '\b0[4-5]\d{7}\b'  # БИК (9 digits, starts with 04 or 05)
      context: ["бик", "БИК", "банковский идентификационный код"]
      confidence: 0.85
    
    # Russian bank account numbers - from research
    - name: "RussianBankAccountRecognizer"
      entity: "BANK_ACCOUNT_RU"
      patterns:
        - '\b40[5-8]\d{17}\b'  # Settlement account (20 digits)
        - '\b301\d{17}\b'  # Correspondent account (20 digits)
      context: ["расчетный счет", "р/с", "корреспондентский счет", "к/с"]
      confidence: 0.8
    
    # Russian OMS Policy (Medical Insurance) - from research
    - name: "RussianOmsPolicyRecognizer"
      entity: "OMS_POLICY_RU"
      patterns:
        - '\b\d{16}\b'  # New format (16 digits)
        - '\b\d{9}\b'  # Old format (9 digits)
      context: ["омс", "ОМС", "полис", "медицинский страховой", "страховая медицинская"]
      confidence: 0.8
    
    # Russian Driver's License - from research
    - name: "RussianDriverLicenseRecognizer"
      entity: "DRIVER_LICENSE_RU"
      patterns:
        - '\b\d{2}\s*\d{2}\s*\d{6}\b'  # XX XX XXXXXX format
      context: ["водительское удостоверение", "ВУ", "права", "водительские права"]
      confidence: 0.75
    
    # Russian Birth Certificate - from research
    - name: "RussianBirthCertRecognizer"
      entity: "BIRTH_CERT_RU"
      patterns:
        - '\b[IVX]+[-\s]*[А-ЯЁ]{2}[-\s]*№?\s*\d{6}\b'  # IV-ЖА №123456 format
        - '\b[IVX]+\s+[А-ЯЁ]{2}\s+\d{6}\b'  # IV ЖА 123456 format
      context: ["свидетельство о рождении", "ЗАГС", "рождении"]
      confidence: 0.7
    
    # Russian Military ID - from research
    - name: "RussianMilitaryIdRecognizer"
      entity: "MILITARY_ID_RU"
      patterns:
        - '\b[А-ЯЁ]{2}\s*\d{6,7}\b'  # Series (2 Cyrillic) + number
      context: ["военный билет", "военной", "призыв", "мобилизация"]
      confidence: 0.75
    
    # Russian VIN (Vehicle ID) - from research
    - name: "RussianVinRecognizer"
      entity: "VIN_RU"
      patterns:
        - '\b[A-HJ-NPR-Z0-9]{17}\b'  # 17 alphanumeric (no I, O, Q)
      context: ["вин", "VIN", "идентификационный номер тс", "номер кузова"]
      confidence: 0.85
    
    # Russian OKATO/OKTMO (Territorial codes) - from research
    - name: "RussianOkatoRecognizer"
      entity: "OKATO_RU"
      patterns:
        - '\b\d{8}\b'  # OKATO (8 digits)
        - '\b\d{11}\b'  # OKTMO (11 digits)
      context: ["окато", "ОКАТО", "октмо", "ОКТМО", "территориальный код"]
      confidence: 0.6
    
    # Russian car registration numbers
    - name: "RussianCarNumberRecognizer"
      entity: "CAR_NUMBER_RU"
      patterns:
        - '\b[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}\b'  # А777АА777
      context: ["номер", "машина", "автомобиль", "государственный"]
      confidence: 0.85
    
    # Russian age patterns
    - name: "RussianAgeRecognizer"
      entity: "AGE_RU"
      patterns:
        - 'мне\s+\d{1,2}\s+лет'    # "мне 25 лет"
        - '\b\d{1,2}\s+лет\b'      # "25 лет"
      context: ["лет", "года"]
      confidence: 0.7
    
    # Russian short names (Иванов И.О.)
    - name: "RussianShortNameRecognizer"
      entity: "PERSON_SHORT_RU"
      patterns:
        - '\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.(?:\s*[А-ЯЁ]\.)?'  # Иванов И.О.
        - '\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.'        # Иванов И. О.
      confidence: 0.8
    
    # Corporate sensitive patterns
    - name: "PasswordRecognizer"
      entity: "PASSWORD"
      patterns:
        - '(пароль|password|passwd|pwd)[_\s:=]+[^\s]+'
      case_sensitive: false
      confidence: 0.95
    
    - name: "LoginRecognizer"
      entity: "LOGIN"
      patterns:
        - '(логин|login|username)[_\s:=]+[^\s]+'
      case_sensitive: false
      confidence: 0.95
    
    - name: "ApiKeyRecognizer"
      entity: "API_KEY"
      patterns:
        - '(api[_-]?key|apikey)[_\s:=]+[^\s]+'
        - '(ключ[а-яё]*\s*(api|доступа))[_\s:=]+[^\s]+'
      case_sensitive: false
      confidence: 0.95
    
    - name: "InternalUrlRecognizer"
      entity: "INTERNAL_URL"
      patterns:
        - '(http|https)://(intranet|internal|local|192\.168|10\.|172\.(1[6-9]|2|3[01]))[^\s]*'
        - 'www\.(intranet|internal|local)[^\s]*'
      case_sensitive: false
      confidence: 0.9

# Stage 2: EU-PII-Safeguard
stage2_eu_pii:
  enabled: true
  
  # Model configuration
  model:
    # HuggingFace model ID
    name: "tabularisai/eu-pii-safeguard"
    
    # Device for inference
    # Options: auto, cpu, cuda, cuda:0, etc.
    device: "auto"
    
    # Batch size for inference
    batch_size: 8
    
    # Minimum confidence threshold (0.0-1.0)
    confidence_threshold: 0.5
  
  # Inference provider
  provider:
    # Options: local (torch), vllm, mosec, openai
    type: "local"
    
    # vLLM settings (if provider.type = vllm)
    vllm:
      url: "http://localhost:8000/v1"
      model: "tabularisai/eu-pii-safeguard"
      api_key: "EMPTY"
    
    # MOSEC settings (if provider.type = mosec)
    mosec:
      endpoint: "http://localhost:8000/v1/chat/completions"
      model: "tabularisai/eu-pii-safeguard"
  
  # Entity types to detect (subset of 42 supported)
  # Leave empty to detect all
  entity_types: []

# Stage 3: Russian NER
stage3_ner:
  enabled: true
  
  # Model configuration
  model:
    # Model choice: bert-base (Gherman) or natasha
    name: "Gherman/bert-base-NER-Russian"
    
    # Natasha configuration (if using natasha)
    natasha:
      enabled: false
      # NERette or Miex providers
      provider: "nerette"
    
    device: "auto"
    batch_size: 8
    confidence_threshold: 0.5
  
  # Inference provider
  provider:
    type: "local"  # local, vllm, mosec
  
  # Entity mapping (NER output → placeholder)
  entity_mapping:
    FIRST_NAME: "NAME"
    MIDDLE_NAME: "NAME"
    LAST_NAME: "NAME"
    PERSON: "NAME"
    CITY: "ADDRESS"
    COUNTRY: "ADDRESS"
    REGION: "ADDRESS"
    DISTRICT: "ADDRESS"
    STREET: "ADDRESS"
    ADDRESS: "ADDRESS"
    ORGANIZATION: "COMPANY"

# Deanonymization settings
deanonymization:
  # Handle cases where LLM uses placeholders in output
  # Options: restore_only (only restore from mapping), 
  #          full (restore + preserve unknown placeholders)
  mode: "restore_only"
  
  # Whether to detect and handle nested placeholders
  handle_nested: true
  
  # Placeholder prefix for detection
  placeholder_prefix: "["

# Logging and debugging
logging:
  # Log anonymization operations
  enabled: true
  
  # Log level: DEBUG, INFO, WARNING, ERROR
  level: "INFO"
  
  # Include sample mappings in logs (be careful with PII!)
  include_mappings: false
```

#### 2.2.2 Environment Variables: `.env`

```bash
# =============================================================================
# ANONYMIZATION PIPELINE (Opt-in)
# =============================================================================

# [REQUIRED] Enable/disable anonymization pipeline
ANONYMIZER_ENABLED=true

# Pipeline settings
ANONYMIZER_DETECTION_LEVEL=balanced  # conservative | balanced | aggressive
ANONYMIZER_MODE=cascade              # cascade | parallel

# Per-stage enable/disable (optional, can also be in YAML)
ANONYMIZER_STAGE1_ENABLED=true
ANONYMIZER_STAGE2_ENABLED=true
ANONYMIZER_STAGE3_ENABLED=true

# Stage 2: EU-PII-Safeguard (requires HuggingFace token acceptance)
ANONYMIZER_EU_PII_PROVIDER=local    # local | vllm | mosec

# Stage 3: Russian NER
ANONYMIZER_RUSSIAN_NER_PROVIDER=local  # local | vllm | mosec

# Inference endpoints (if using vllm/mosec)
EU_PII_VLLM_URL=http://localhost:8000/v1
EU_PII_VLLM_MODEL=tabularisai/eu-pii-safeguard
RUSSIAN_NER_VLLM_URL=http://localhost:8000/v1
RUSSIAN_NER_VLLM_MODEL=Gherman/bert-base-NER-Russian

# Session mapping TTL (seconds) - for chat UI multi-turn
ANONYMIZER_SESSION_MAPPING_TTL=3600
```

#### 2.2.3 Settings Integration: `rag_engine/config/settings.py`

Add to `Settings` class:

```python
# Anonymization Settings
anonymizer_enabled: bool
anonymizer_detection_level: str
anonymizer_mode: str
anonymizer_stage1_enabled: bool
anonymizer_stage2_enabled: bool
anonymizer_stage3_enabled: bool
anonymizer_eu_pii_provider: str
anonymizer_russian_ner_provider: str
eu_pii_vllm_url: str
eu_pii_vllm_model: str
russian_ner_vllm_url: str
russian_ner_vllm_model: str
anonymizer_session_mapping_ttl: int
```

---

## 3. Core Implementation

### 3.1 Data Types (`rag_engine/anonymization/types.py`)

```python
"""Data types for anonymization pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DetectionStage(Enum):
    """Stage at which entity was detected."""
    STAGE1_REGEX = 1
    STAGE2_EU_PII = 2
    STAGE3_RUSSIAN_NER = 3


class EntityType(Enum):
    """Supported PII entity types."""
    # Basic
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"
    
    # Russian documents
    PASSPORT = "PASSPORT"
    SNILS = "SNILS"
    INN = "INN"
    OGRN = "OGRN"           # State Registration Number (13 digits)
    KPP = "KPP"             # Tax Registration Reason Code (9 digits)
    OKPO = "OKPO"           # National Classifier of Businesses (8-10 digits)
    OKVED = "OKVED"         # Types of Economic Activity
    
    # Financial
    BANK_CARD = "BANK_CARD"
    BIC = "BIC"             # Bank Identification Code (9 digits)
    BANK_ACCOUNT = "BANK_ACCOUNT"  # Settlement/Correspondent account
    
    # Additional Russian identifiers (from research)
    OMS_POLICY = "OMS_POLICY"     # Medical insurance policy
    DRIVER_LICENSE = "DRIVER_LICENSE"  # Driver's license
    BIRTH_CERT = "BIRTH_CERT"     # Birth certificate
    MILITARY_ID = "MILITARY_ID"   # Military ID
    VIN = "VIN"                   # Vehicle Identification Number
    OKATO = "OKATO"               # Territorial codes
    
    # Personal
    NAME = "NAME"
    AGE = "AGE"
    BIRTHDATE = "BIRTHDATE"
    
    # Location
    ADDRESS = "ADDRESS"
    STREET_ADDRESS = "STREET_ADDRESS"
    BUILDING_NUMBER = "BUILDING_NUMBER"
    CITY = "CITY"
    COUNTRY = "COUNTRY"
    
    # Organization
    COMPANY = "COMPANY"
    ORGANIZATION = "ORGANIZATION"
    
    # Corporate sensitive
    PASSWORD = "PASSWORD"
    LOGIN = "LOGIN"
    API_KEY = "API_KEY"
    INTERNAL_URL = "INTERNAL_URL"
    
    # Vehicle
    CAR_NUMBER = "CAR_NUMBER"
    
    # Other
    DATE = "DATE"


# =============================================================================
# Semantic Placeholder Mapping
# =============================================================================
# Based on: https://github.com/levitation-opensource/DataAnonymiser
# More readable for LLM than generic placeholders like [NAME_0]
# =============================================================================

ENTITY_TO_SEMANTIC_PLACEHOLDER: dict[EntityType, str] = {
    EntityType.NAME: "Person",
    EntityType.EMAIL: "Email",
    EntityType.PHONE: "Phone",
    EntityType.URL: "Url",
    EntityType.IP_ADDRESS: "IP Address",
    EntityType.PASSPORT: "Passport",
    EntityType.SNILS: "SNILS",
    EntityType.INN: "INN",
    EntityType.OGRN: "OGRN",
    EntityType.KPP: "KPP",
    EntityType.OKPO: "OKPO",
    EntityType.OKVED: "OKVED",
    EntityType.BANK_CARD: "Card",
    EntityType.BIC: "BIC",
    EntityType.BANK_ACCOUNT: "Bank Account",
    EntityType.OMS_POLICY: "Medical Policy",
    EntityType.DRIVER_LICENSE: "Driver License",
    EntityType.BIRTH_CERT: "Birth Certificate",
    EntityType.MILITARY_ID: "Military ID",
    EntityType.VIN: "VIN",
    EntityType.OKATO: "Territorial Code",
    EntityType.AGE: "Age",
    EntityType.BIRTHDATE: "Date",
    EntityType.ADDRESS: "Address",
    EntityType.STREET_ADDRESS: "Street",
    EntityType.BUILDING_NUMBER: "Building",
    EntityType.CITY: "City",
    EntityType.COUNTRY: "Country",
    EntityType.COMPANY: "Organization",
    EntityType.ORGANIZATION: "Organization",
    EntityType.PASSWORD: "Password",
    EntityType.LOGIN: "Login",
    EntityType.API_KEY: "API Key",
    EntityType.INTERNAL_URL: "Internal URL",
    EntityType.CAR_NUMBER: "Vehicle",
    EntityType.DATE: "Date",
}


@dataclass
class DetectedEntity:
    """A detected PII entity."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float
    stage: DetectionStage
    
    def to_placeholder(self, counter: int) -> str:
        """Generate placeholder for this entity."""
        return f"[{self.entity_type.value}_{counter}]"


@dataclass
class AnonymizationResult:
    """Result of anonymization operation."""
    original_text: str
    anonymized_text: str
    entities: list[DetectedEntity] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)
    
    @property
    def has_entities(self) -> bool:
        return len(self.entities) > 0


@dataclass
class DeanonymizationResult:
    """Result of deanonymization operation."""
    anonymized_text: str
    deanonymized_text: str
    restored_count: int
    remaining_placeholders: list[str] = field(default_factory=list)
```

### 3.2 Base Stage Interface (`rag_engine/anonymization/stages/base.py`)

```python
"""Base interface for anonymization stages."""

from abc import ABC, abstractmethod

from rag_engine.anonymization.types import AnonymizationResult, DetectedEntity


class AnonymizationStage(ABC):
    """Base class for anonymization pipeline stages."""
    
    def __init__(self, config: dict[str, Any]):
        """Initialize stage with configuration."""
        self.config = config
        self.enabled = config.get("enabled", True)
    
    @abstractmethod
    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect PII entities in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities with positions and types
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if stage is enabled."""
        return self.enabled
```

### 3.3 Stage 1: Presidio + Custom Russian Recognizers (`rag_engine/anonymization/stages/stage1_regex.py`)

```python
"""Stage 1: Presidio orchestrator with custom Russian recognizers.

This stage uses Microsoft Presidio as the orchestrator, extended with custom
Russian-specific recognizers from rus-anonymizer and ru-smb-pd-anonymizer.

Architecture:
- Presidio built-in recognizers for Latin/English patterns
- Custom PatternRecognizer subclasses for Russian-specific PII
"""

import logging
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    PhoneRecognizer,
    CreditCardRecognizer,
    UrlRecognizer,
    IpRecognizer,
    PatternRecognizer,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from rag_engine.anonymization.stages.base import AnonymizationStage
from rag_engine.anonymization.types import (
    DetectionStage,
    DetectedEntity,
    EntityType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Russian Recognizers
# =============================================================================

class RussianPhoneRecognizer(PatternRecognizer):
    """Russian phone number recognizer.
    
    Patterns from rus-anonymizer:
    - +7 (XXX) XXX-XX-XX
    - +7 XXX XXX-XX-XX
    - 8 (XXX) XXX-XX-XX
    - 8 XXX XXX-XX-XX
    - +7XXXXXXXXXX
    - 8XXXXXXXXXX
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', 0.9),
            Pattern(r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', 0.9),
            Pattern(r'8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', 0.9),
            Pattern(r'8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}', 0.9),
            Pattern(r'\+7\d{10}', 0.85),
            Pattern(r'8\d{10}', 0.85),
        ]
        super().__init__(
            supported_entity="PHONE_RU",
            patterns=patterns,
            context=["телефон", "тел", "мобильный", "контактный", "связи"],
        )


class RussianPassportRecognizer(PatternRecognizer):
    """Russian passport number recognizer (series + number)."""
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{2}\s+\d{2}\s+\d{6}\b', 0.85),  # XX XX XXXXXX
            Pattern(r'\b\d{4}\s+\d{6}\b', 0.85),          # XXXX XXXXXX
        ]
        super().__init__(
            supported_entity="PASSPORT_RU",
            patterns=patterns,
            context=["паспорт", "паспорта", "паспортом", "свид-во"],
        )


class RussianSnilsRecognizer(PatternRecognizer):
    """Russian SNILS (insurance number) recognizer."""
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b', 0.9),
        ]
        super().__init__(
            supported_entity="SNILS_RU",
            patterns=patterns,
            context=["снилс", "СНИЛС", "страховой", "страхования"],
        )


class RussianInnRecognizer(PatternRecognizer):
    """Russian INN (tax ID) recognizer."""
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{10}\b', 0.8),  # Legal entity (10 digits)
            Pattern(r'\b\d{12}\b', 0.8),  # Individual (12 digits)
        ]
        super().__init__(
            supported_entity="INN_RU",
            patterns=patterns,
            context=["инн", "ИНН", "налоговый", "налоговая"],
        )


class RussianOgrnRecognizer(PatternRecognizer):
    """Russian OGRN (State Registration Number) recognizer.
    
    OGRN - Primary State Registration Number (Основной государственный регистрационный номер)
    - 13 digits: Legal entities
    - 15 digits: Individual entrepreneurs (ОГРНИП)
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{13}\b', 0.85),  # OGRN (legal entity)
            Pattern(r'\b\d{15}\b', 0.85),  # OGRNIP (individual entrepreneur)
        ]
        super().__init__(
            supported_entity="OGRN_RU",
            patterns=patterns,
            context=["огрн", "ОГРН", "огрнип", "ОГРНИП", "госрегистрация", "регистрация"],
        )


class RussianKppRecognizer(PatternRecognizer):
    """Russian KPP (Tax Registration Reason Code) recognizer.
    
    КПП - Code of Reason for Registration (Код причины постановки на учёт)
    - 9 digits
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{9}\b', 0.8),
        ]
        super().__init__(
            supported_entity="KPP_RU",
            patterns=patterns,
            context=["кпп", "КПП", "причина постановки"],
        )


class RussianOkpoRecognizer(PatternRecognizer):
    """Russian OKPO (National Classifier of Businesses) recognizer.
    
    ОКПО - Russian National Classifier of Businesses and Organizations
    - 8 digits: Legal entities
    - 10 digits: Individual entrepreneurs
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{8}\b', 0.7),   # ОКПО legal
            Pattern(r'\b\d{10}\b', 0.7),  # ОКПО IE
        ]
        super().__init__(
            supported_entity="OKPO_RU",
            patterns=patterns,
            context=["окпо", "ОКПО", "классификатор"],
        )


class RussianOkvedRecognizer(PatternRecognizer):
    """Russian OKVED (Types of Economic Activity) recognizer.
    
    ОКВЭД - Russian National Classifier of Types of Economic Activity
    - Format: XX.XX.XX or X.XX.XX
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{2}\.\d{2}\.\d{2}\b', 0.7),
            Pattern(r'\b\d{1,2}\.\d{2}\.\d{2}\b', 0.7),
        ]
        super().__init__(
            supported_entity="OKVED_RU",
            patterns=patterns,
            context=["оквэд", "ОКВЭД", "вид деятельности", "экономическая деятельность"],
        )


class RussianBicRecognizer(PatternRecognizer):
    """Russian BIC (Bank Identification Code) recognizer.
    
    БИК - Bank Identification Code (Банковский идентификационный код)
    - 9 digits, starts with 04 or 05
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b0[4-5]\d{7}\b', 0.85),
        ]
        super().__init__(
            supported_entity="BIC_RU",
            patterns=patterns,
            context=["бик", "БИК", "банковский идентификационный код"],
        )


class RussianBankAccountRecognizer(PatternRecognizer):
    """Russian bank account number recognizer.
    
    Settlement account (Расчетный счет): 20 digits, starts with 405-408
    Correspondent account (Корреспондентский счет): 20 digits, starts with 301
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b40[5-8]\d{17}\b', 0.8),  # Settlement account
            Pattern(r'\b301\d{17}\b', 0.8),       # Correspondent account
        ]
        super().__init__(
            supported_entity="BANK_ACCOUNT_RU",
            patterns=patterns,
            context=["расчетный счет", "р/с", "корреспондентский счет", "к/с", "счет"],
        )


class RussianOmsPolicyRecognizer(PatternRecognizer):
    """Russian OMS (Mandatory Medical Insurance) policy recognizer.
    
    ОМС - Polis of Mandatory Medical Insurance
    - 16 digits: New format
    - 9 digits: Old format
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{16}\b', 0.8),  # New format
            Pattern(r'\b\d{9}\b', 0.7),   # Old format
        ]
        super().__init__(
            supported_entity="OMS_POLICY_RU",
            patterns=patterns,
            context=["омс", "ОМС", "полис", "медицинский страховой", "страховая медицинская"],
        )


class RussianDriverLicenseRecognizer(PatternRecognizer):
    """Russian Driver's License recognizer.
    
    Водительское удостоверение
    - Format: XX XX XXXXXX (10 digits with spaces)
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{2}\s*\d{2}\s*\d{6}\b', 0.75),
        ]
        super().__init__(
            supported_entity="DRIVER_LICENSE_RU",
            patterns=patterns,
            context=["водительское удостоверение", "ВУ", "права", "водительские права"],
        )


class RussianBirthCertRecognizer(PatternRecognizer):
    """Russian Birth Certificate recognizer.
    
    Свидетельство о рождении
    - Format: Roman numeral + 2 Cyrillic letters + 6 digits
      Example: IV-ЖА-123456 or IV ЖА 123456
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b[IVX]+[-\s]*[А-ЯЁ]{2}[-\s]*№?\s*\d{6}\b', 0.7),
            Pattern(r'\b[IVX]+\s+[А-ЯЁ]{2}\s+\d{6}\b', 0.7),
        ]
        super().__init__(
            supported_entity="BIRTH_CERT_RU",
            patterns=patterns,
            context=["свидетельство о рождении", "ЗАГС", "рождении"],
        )


class RussianMilitaryIdRecognizer(PatternRecognizer):
    """Russian Military ID recognizer.
    
    Военный билет
    - Format: 2 Cyrillic letters + 6-7 digits
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b[А-ЯЁ]{2}\s*\d{6,7}\b', 0.75),
        ]
        super().__init__(
            supported_entity="MILITARY_ID_RU",
            patterns=patterns,
            context=["военный билет", "военной", "призыв", "мобилизация"],
        )


class RussianVinRecognizer(PatternRecognizer):
    """Russian VIN (Vehicle Identification Number) recognizer.
    
    VIN - Идентификационный номер транспортного средства
    - 17 alphanumeric characters
    - Excludes I, O, Q (to avoid confusion)
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b[A-HJ-NPR-Z0-9]{17}\b', 0.85),
        ]
        super().__init__(
            supported_entity="VIN_RU",
            patterns=patterns,
            context=["вин", "VIN", "идентификационный номер тс", "номер кузова", "номер шасси"],
        )


class RussianOkatoRecognizer(PatternRecognizer):
    """Russian OKATO/OKTMO (Territorial codes) recognizer.
    
    ОКАТО/ОКТМО - Russian classifiers of administrative-territorial divisions
    - OKATO: 8 digits
    - OKTMO: 11 digits
    """
    
    def __init__(self):
        patterns = [
            Pattern(r'\b\d{8}\b', 0.6),   # OKATO
            Pattern(r'\b\d{11}\b', 0.6),  # OKTMO
        ]
        super().__init__(
            supported_entity="OKATO_RU",
            patterns=patterns,
            context=["окато", "ОКАТО", "октмо", "ОКТМО", "территориальный код"],
        )


class RussianCarNumberRecognizer(PatternRecognizer):
    """Russian car registration number recognizer."""
    
    def __init__(self):
        patterns = [
            Pattern(r'\b[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}\b', 0.85),
        ]
        super().__init__(
            supported_entity="CAR_NUMBER_RU",
            patterns=patterns,
            context=["номер", "машина", "автомобиль", "государственный", "регистрация"],
        )


class RussianAgeRecognizer(PatternRecognizer):
    """Russian age pattern recognizer."""
    
    def __init__(self):
        patterns = [
            Pattern(r'мне\s+\d{1,2}\s+лет', 0.7),
            Pattern(r'\b\d{1,2}\s+лет\b', 0.6),
        ]
        super().__init__(
            supported_entity="AGE_RU",
            patterns=patterns,
            context=["лет", "года"],
        )


class RussianShortNameRecognizer(PatternRecognizer):
    """Russian short name/initials recognizer (Иванов И.О.)."""
    
    def __init__(self):
        patterns = [
            Pattern(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.(?:\s*[А-ЯЁ]\.)?', 0.8),
            Pattern(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.', 0.8),
        ]
        super().__init__(
            supported_entity="PERSON_SHORT_RU",
            patterns=patterns,
        )


class PasswordRecognizer(PatternRecognizer):
    """Corporate password detector."""
    
    def __init__(self):
        patterns = [
            Pattern(r'(пароль|password|passwd|pwd)[_\s:=]+[^\s]+', 0.95),
        ]
        super().__init__(
            supported_entity="PASSWORD",
            patterns=patterns,
        )


class LoginRecognizer(PatternRecognizer):
    """Corporate login/username detector."""
    
    def __init__(self):
        patterns = [
            Pattern(r'(логин|login|username)[_\s:=]+[^\s]+', 0.95),
        ]
        super().__init__(
            supported_entity="LOGIN",
            patterns=patterns,
        )


class ApiKeyRecognizer(PatternRecognizer):
    """API key/token detector."""
    
    def __init__(self):
        patterns = [
            Pattern(r'(api[_-]?key|apikey)[_\s:=]+[^\s]+', 0.95),
            Pattern(r'(ключ[а-яё]*\s*(api|доступа))[_\s:=]+[^\s]+', 0.95),
        ]
        super().__init__(
            supported_entity="API_KEY",
            patterns=patterns,
        )


class InternalUrlRecognizer(PatternRecognizer):
    """Internal/corporate URL detector."""
    
    def __init__(self):
        patterns = [
            Pattern(r'(http|https)://(intranet|internal|local|192\.168|10\.|172\.(1[6-9]|2|3[01]))[^\s]*', 0.9),
            Pattern(r'www\.(intranet|internal|local)[^\s]*', 0.9),
        ]
        super().__init__(
            supported_entity="INTERNAL_URL",
            patterns=patterns,
        )


# =============================================================================
# Stage Implementation
# =============================================================================

class RegexAnonymizationStage(AnonymizationStage):
    """Stage 1: Presidio orchestrator with custom Russian recognizers.
    
    Uses Presidio as the main framework, extended with custom recognizers
    for Russian-specific PII types.
    """
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
        # Build registry with custom recognizers
        registry = RecognizerRegistry()
        
        # Load built-in recognizers (configurable)
        built_in = config.get("presidio", {}).get("built_in_entities", [
            "EMAIL_ADDRESS", "CREDIT_CARD", "URL", "IP_ADDRESS", "DATE_TIME"
        ])
        
        if "EMAIL_ADDRESS" in built_in:
            registry.add_recognizer(EmailRecognizer())
        if "CREDIT_CARD" in built_in:
            registry.add_recognizer(CreditCardRecognizer())
        if "URL" in built_in:
            registry.add_recognizer(UrlRecognizer())
        if "IP_ADDRESS" in built_in:
            registry.add_recognizer(IpRecognizer())
        # Note: PhoneRecognizer is generic, we use RussianPhoneRecognizer instead
        
        # Add custom Russian recognizers
        custom_recognizers = config.get("custom_recognizers", {}).get("enabled", True)
        if custom_recognizers:
            registry.add_recognizer(RussianPhoneRecognizer())
            registry.add_recognizer(RussianPassportRecognizer())
            registry.add_recognizer(RussianSnilsRecognizer())
            registry.add_recognizer(RussianInnRecognizer())
            registry.add_recognizer(RussianOgrnRecognizer())
            registry.add_recognizer(RussianKppRecognizer())
            registry.add_recognizer(RussianOkpoRecognizer())
            registry.add_recognizer(RussianOkvedRecognizer())
            registry.add_recognizer(RussianBicRecognizer())
            registry.add_recognizer(RussianBankAccountRecognizer())
            registry.add_recognizer(RussianOmsPolicyRecognizer())
            registry.add_recognizer(RussianDriverLicenseRecognizer())
            registry.add_recognizer(RussianBirthCertRecognizer())
            registry.add_recognizer(RussianMilitaryIdRecognizer())
            registry.add_recognizer(RussianVinRecognizer())
            registry.add_recognizer(RussianOkatoRecognizer())
            registry.add_recognizer(RussianCarNumberRecognizer())
            registry.add_recognizer(RussianAgeRecognizer())
            registry.add_recognizer(RussianShortNameRecognizer())
            registry.add_recognizer(PasswordRecognizer())
            registry.add_recognizer(LoginRecognizer())
            registry.add_recognizer(ApiKeyRecognizer())
            registry.add_recognizer(InternalUrlRecognizer())
        
        # Initialize Presidio analyzer with custom registry
        self.analyzer = AnalyzerEngine(registry=registry, supported_languages=["en", "ru"])
        self.anonymizer = AnonymizerEngine()
        
        logger.info("Stage 1 initialized with Presidio + custom Russian recognizers")
    
    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect PII using Presidio with custom recognizers."""
        entities = []
        
        # Run Presidio analyzer
        results = self.analyzer.analyze(
            text=text,
            language="ru",  # Try Russian first for mixed text
        )
        
        for result in results:
            entity_type = self._map_entity_type(result.entity_type)
            if entity_type:
                entities.append(DetectedEntity(
                    text=text[result.start:result.end],
                    entity_type=entity_type,
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                    stage=DetectionStage.STAGE1_REGEX,
                ))
        
        return entities
    
    def _map_entity_type(self, presidio_type: str) -> EntityType | None:
        """Map Presidio entity types to our EntityType enum."""
        mapping = {
            # Presidio built-in
            "EMAIL_ADDRESS": EntityType.EMAIL,
            "CREDIT_CARD": EntityType.BANK_CARD,
            "URL": EntityType.URL,
            "IP_ADDRESS": EntityType.IP_ADDRESS,
            "DATE_TIME": EntityType.DATE,
            
            # Custom Russian
            "PHONE_RU": EntityType.PHONE,
            "PASSPORT_RU": EntityType.PASSPORT,
            "SNILS_RU": EntityType.SNILS,
            "INN_RU": EntityType.INN,
            "OGRN_RU": EntityType.OGRN,
            "KPP_RU": EntityType.KPP,
            "OKPO_RU": EntityType.OKPO,
            "OKVED_RU": EntityType.OKVED,
            "BIC_RU": EntityType.BIC,
            "BANK_ACCOUNT_RU": EntityType.BANK_ACCOUNT,
            "OMS_POLICY_RU": EntityType.OMS_POLICY,
            "DRIVER_LICENSE_RU": EntityType.DRIVER_LICENSE,
            "BIRTH_CERT_RU": EntityType.BIRTH_CERT,
            "MILITARY_ID_RU": EntityType.MILITARY_ID,
            "VIN_RU": EntityType.VIN,
            "OKATO_RU": EntityType.OKATO,
            "CAR_NUMBER_RU": EntityType.CAR_NUMBER,
            "AGE_RU": EntityType.AGE,
            "PERSON_SHORT_RU": EntityType.NAME,
            
            # Corporate
            "PASSWORD": EntityType.PASSWORD,
            "LOGIN": EntityType.LOGIN,
            "API_KEY": EntityType.API_KEY,
            "INTERNAL_URL": EntityType.INTERNAL_URL,
        }
        return mapping.get(presidio_type.upper())
```

### 3.4 Stage 2: EU-PII-Safeguard (`rag_engine/anonymization/stages/stage2_eu_pii.py`)

```python
"""Stage 2: EU-PII-Safeguard multilingual PII detection."""

import logging
from typing import Any

from rag_engine.anonymization.stages.base import AnonymizationStage
from rag_engine.anonymization.types import (
    DetectionStage,
    DetectedEntity,
    EntityType,
)

logger = logging.getLogger(__name__)


class EUPIIAnonymizationStage(AnonymizationStage):
    """Stage 2: EU-PII-Safeguard PII detection.
    
    Detects 42 entity types across 26 European languages.
    Uses tabularisai/eu-pii-safeguard model.
    """
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
        self.model_config = config.get("model", {})
        self.provider_config = config.get("provider", {})
        
        # Lazy load model
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load the model pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            
            model_name = self.model_config.get("name", "tabularisai/eu-pii-safeguard")
            device = self.model_config.get("device", "auto")
            
            # Use provider factory to get inference
            # For now, use local inference
            self._pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=device,
            )
        
        return self._pipeline
    
    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect PII using EU-PII-Safeguard."""
        if not self.is_enabled():
            return []
        
        try:
            results = self.pipeline(text)
        except Exception as e:
            logger.warning(f"EU-PII-Safeguard inference failed: {e}")
            return []
        
        entities = []
        for result in results:
            entity_type = self._map_eu_pii_entity(result.get("entity_group"))
            if entity_type:
                entities.append(DetectedEntity(
                    text=result.get("word", ""),
                    entity_type=entity_type,
                    start=result.get("start", 0),
                    end=result.get("end", 0),
                    confidence=result.get("score", 0.0),
                    stage=DetectionStage.STAGE2_EU_PII,
                ))
        
        return entities
    
    def _map_eu_pii_entity(self, eu_pii_type: str | None) -> EntityType | None:
        """Map EU-PII entity types to our EntityType enum."""
        if not eu_pii_type:
            return None
        
        # EU-PII-Safeguard entity types (subset)
        mapping = {
            # Person
            "FIRST_NAME": EntityType.NAME,
            "LAST_NAME": EntityType.NAME,
            "FULL_NAME": EntityType.NAME,
            
            # Location
            "ADDRESS": EntityType.ADDRESS,
            "STREET": EntityType.STREET_ADDRESS,
            "CITY": EntityType.CITY,
            "COUNTRY": EntityType.COUNTRY,
            
            # Organization
            "ORGANIZATION": EntityType.COMPANY,
            "COMPANY": EntityType.COMPANY,
            
            # Contact
            "EMAIL": EntityType.EMAIL,
            "PHONE": EntityType.PHONE,
            "URL": EntityType.URL,
            
            # Documents
            "PASSPORT": EntityType.PASSPORT,
            "ID_CARD": EntityType.PASSPORT,
            "NATIONAL_ID": EntityType.PASSPORT,
            
            # Financial
            "BANK_ACCOUNT": EntityType.BANK_CARD,
            "CREDIT_CARD": EntityType.BANK_CARD,
            
            # Other
            "DATE": EntityType.DATE,
            "IP_ADDRESS": EntityType.IP_ADDRESS,
        }
        
        return mapping.get(eu_pii_type.upper())
```

### 3.5 Stage 3: Russian NER (`rag_engine/anonymization/stages/stage3_ner.py`)

```python
"""Stage 3: Russian NER using Gherman/bert-base-NER-Russian."""

import logging
from typing import Any

from rag_engine.anonymization.stages.base import AnonymizationStage
from rag_engine.anonymization.types import (
    DetectionStage,
    DetectedEntity,
    EntityType,
)

logger = logging.getLogger(__name__)


class RussianNERAnonymizationStage(AnonymizationStage):
    """Stage 3: Russian-specific NER.
    
    Uses Gherman/bert-base-NER-Russian for Russian-specific entity detection.
    Complements EU-PII-Safeguard with better Russian person/organization detection.
    """
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        
        self.model_config = config.get("model", {})
        self.entity_mapping = config.get("entity_mapping", {
            "FIRST_NAME": "NAME",
            "MIDDLE_NAME": "NAME",
            "LAST_NAME": "NAME",
            "PERSON": "NAME",
            "CITY": "ADDRESS",
            "COUNTRY": "ADDRESS",
            "REGION": "ADDRESS",
            "DISTRICT": "ADDRESS",
            "STREET": "ADDRESS",
            "ADDRESS": "ADDRESS",
            "ORGANIZATION": "COMPANY",
        })
        
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load the model pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            
            model_name = self.model_config.get("name", "Gherman/bert-base-NER-Russian")
            device = self.model_config.get("device", "auto")
            
            self._pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=device,
            )
        
        return self._pipeline
    
    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect Russian entities using NER."""
        if not self.is_enabled():
            return []
        
        try:
            results = self.pipeline(text)
        except Exception as e:
            logger.warning(f"Russian NER inference failed: {e}")
            return []
        
        entities = []
        for result in results:
            entity_group = result.get("entity_group")
            mapped_type = self.entity_mapping.get(entity_group)
            
            if mapped_type:
                try:
                    entity_type = EntityType[mapped_type]
                except KeyError:
                    entity_type = EntityType.NAME  # Default fallback
                
                entities.append(DetectedEntity(
                    text=result.get("word", ""),
                    entity_type=entity_type,
                    start=result.get("start", 0),
                    end=result.get("end", 0),
                    confidence=result.get("score", 0.0),
                    stage=DetectionStage.STAGE3_RUSSIAN_NER,
                ))
        
        return entities
```

### 3.6 Main Pipeline (`rag_engine/anonymization/pipeline.py`)

```python
"""Main anonymization pipeline orchestrator."""

import logging
import uuid
from typing import Any

from rag_engine.anonymization.config import AnonymizationConfig
from rag_engine.anonymization.mapping_store import MappingStore
from rag_engine.anonymization.stages.stage1_regex import RegexAnonymizationStage
from rag_engine.anonymization.stages.stage2_eu_pii import EUPIIAnonymizationStage
from rag_engine.anonymization.stages.stage3_ner import RussianNERAnonymizationStage
from rag_engine.anonymization.types import (
    AnonymizationResult,
    DeanonymizationResult,
    DetectedEntity,
)

logger = logging.getLogger(__name__)


class AnonymizationPipeline:
    """Cascaded reversible anonymization pipeline.
    
    Runs multiple stages in sequence to maximize PII detection coverage.
    Maintains a mapping for reversible deanonymization.
    """
    
    def __init__(self, config: AnonymizationConfig | None = None):
        """Initialize pipeline with configuration."""
        self.config = config or AnonymizationConfig()
        
        # Initialize stages
        self.stages = []
        
        if self.config.stage1_enabled:
            self.stages.append(RegexAnonymizationStage(self.config.stage1_config))
        
        if self.config.stage2_enabled:
            self.stages.append(EUPIIAnonymizationStage(self.config.stage2_config))
        
        if self.config.stage3_enabled:
            self.stages.append(RussianNERAnonymizationStage(self.config.stage3_config))
        
        # Initialize mapping store
        self.mapping_store = MappingStore(ttl=self.config.session_mapping_ttl)
        
        logger.info(f"AnonymizationPipeline initialized with {len(self.stages)} stages")
    
    def anonymize(
        self, 
        text: str, 
        session_id: str | None = None,
    ) -> tuple[str, dict]:
        """Anonymize text and return (anonymized_text, mapping).
        
        Args:
            text: Input text to anonymize
            session_id: Optional session ID for multi-turn conversations
            
        Returns:
            Tuple of (anonymized_text, pii_mapping)
        """
        if not text or not text.strip():
            return text, {}
        
        # Detect entities from all stages
        all_entities: list[DetectedEntity] = []
        
        for stage in self.stages:
            if stage.is_enabled():
                try:
                    entities = stage.detect(text)
                    all_entities.extend(entities)
                except Exception as e:
                    logger.warning(f"Stage {stage.__class__.__name__} failed: {e}")
        
        # Deduplicate overlapping entities (keep highest confidence)
        entities = self._deduplicate_entities(all_entities)
        
        if not entities:
            logger.debug("No PII entities detected")
            return text, {}
        
        # Generate placeholders and mapping
        mapping: dict[str, str] = {}
        counter = 0
        
        # Sort by start position (reverse) for replacement
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        anonymized = text
        for entity in sorted_entities:
            placeholder = f"[{entity.entity_type.value}_{counter}]"
            mapping[placeholder] = entity.text
            
            # Replace in text
            anonymized = anonymized[:entity.start] + placeholder + anonymized[entity.end:]
            
            # Adjust positions for subsequent entities
            delta = len(placeholder) - (entity.end - entity.start)
            for e in sorted_entities:
                if e.start > entity.start:
                    e.start += delta
                    e.end += delta
            
            counter += 1
        
        # Store mapping for session (if session_id provided)
        mapping_id = None
        if session_id:
            mapping_id = self.mapping_store.store(session_id, mapping)
        
        logger.info(f"Anonymized {len(entities)} entities, mapping_id={mapping_id}")
        
        return anonymized, mapping
    
    def deanonymize(
        self, 
        text: str, 
        mapping: dict[str, str] | None = None,
        mapping_id: str | None = None,
    ) -> DeanonymizationResult:
        """Restore original PII from anonymized text.
        
        Args:
            text: Anonymized text
            mapping: Direct mapping dict (from anonymize output)
            mapping_id: Mapping ID (for session-based retrieval)
            
        Returns:
            DeanonymizationResult with restored text
        """
        # Get mapping from store if mapping_id provided
        if mapping_id and not mapping:
            mapping = self.mapping_store.get(mapping_id)
        
        if not mapping:
            return DeanonymizationResult(
                anonymized_text=text,
                deanonymized_text=text,
                restored_count=0,
            )
        
        # Sort placeholders by length (reverse) to avoid partial replacements
        sorted_placeholders = sorted(mapping.keys(), key=len, reverse=True)
        
        deanonymized = text
        restored_count = 0
        remaining_placeholders = []
        
        for placeholder in sorted_placeholders:
            if placeholder in deanonymized:
                deanonymized = deanonymized.replace(placeholder, mapping[placeholder])
                restored_count += 1
            else:
                # Check for variations (nested/repeated)
                pass
        
        # Check for remaining placeholders that weren't restored
        import re
        placeholder_pattern = r'\[[A-Z_]+_\d+\]'
        remaining_placeholders = re.findall(placeholder_pattern, deanonymized)
        
        return DeanonymizationResult(
            anonymized_text=text,
            deanonymized_text=deanonymized,
            restored_count=restored_count,
            remaining_placeholders=remaining_placeholders,
        )
    
    def _deduplicate_entities(
        self, 
        entities: list[DetectedEntity]
    ) -> list[DetectedEntity]:
        """Remove overlapping entities, keeping highest confidence."""
        if not entities:
            return []
        
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        
        result = []
        for entity in sorted_entities:
            overlaps = False
            for existing in result:
                if not (entity.end <= existing.start or entity.start >= existing.end):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result
```

### 3.7 LLM Wrapper Prompt (`rag_engine/anonymization/prompts.py`)

Add note to user message wrapper for LLM context:

```python
"""LLM prompts for anonymization context."""

ANONYMIZATION_CONTEXT_PROMPT = """
<anonymization_note>
This conversation has been anonymized for privacy protection. 
The following placeholders have been used:
- [NAME_*] - Personal names
- [EMAIL_*] - Email addresses  
- [PHONE_*] - Phone numbers
- [ADDRESS_*] - Physical addresses
- [COMPANY_*] - Organization/company names
- [PASSPORT_*], [SNILS_*], [INN_*] - Russian document numbers
- [BANK_CARD_*] - Bank card numbers
- [PASSWORD_*], [LOGIN_*], [API_KEY_*] - Corporate credentials
- [CAR_NUMBER_*] - Vehicle registration numbers
- [INTERNAL_URL_*] - Internal/corporate URLs

When responding, you may use these placeholders in your answer if needed.
The placeholders will be automatically restored to original values before the user sees your response.
</anonymization_note>
"""
```

---

## 4. Integration Points

### 4.0 Unified Mapping Store & Helpers

To avoid code duplication between Chat UI and Platform integration paths, use a single abstraction:

#### 4.0.1 Mapping Store (`rag_engine/anonymization/mapping_store.py`)

```python
"""Unified PII mapping storage for both Chat UI and Platform paths."""

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class MappingStore:
    """Unified mapping storage with TTL and optional Redis backend.
    
    Provides single API for both:
    - Chat UI: stores mappings in-memory with session_id + turn_index
    - Platform: returns mapping directly for single-turn processing
    """
    
    def __init__(self, ttl_seconds: int = 3600, redis_url: str | None = None):
        """Initialize mapping store.
        
        Args:
            ttl_seconds: Time-to-live for mappings (default 1 hour)
            redis_url: Optional Redis URL for distributed deployments
        """
        self.ttl = ttl_seconds
        self._memory: dict[str, tuple[dict, float]] = {}  # mapping_id -> (mapping, expiry)
        
        # Redis backend (optional)
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url)
            except ImportError:
                logger.warning("redis not installed, using in-memory store")
    
    def store(self, mapping: dict[str, str]) -> str:
        """Store mapping and return mapping_id.
        
        Args:
            mapping: PII placeholder → original value mapping
            
        Returns:
            Unique mapping ID for later retrieval
        """
        mapping_id = f"anon_{uuid.uuid4().hex[:12]}"
        expiry = time.time() + self.ttl
        
        if self._redis:
            self._redis.setex(
                mapping_id,
                self.ttl,
                json.dumps(mapping)
            )
        else:
            self._memory[mapping_id] = (mapping, expiry)
        
        logger.debug(f"Stored mapping {mapping_id} with {len(mapping)} entries")
        return mapping_id
    
    def get(self, mapping_id: str) -> dict[str, str] | None:
        """Retrieve mapping by ID.
        
        Args:
            mapping_id: Mapping ID from store()
            
        Returns:
            Mapping dict or None if not found/expired
        """
        if self._redis:
            data = self._redis.get(mapping_id)
            if data:
                return json.loads(data)
            return None
        else:
            entry = self._memory.get(mapping_id)
            if not entry:
                return None
            
            mapping, expiry = entry
            if time.time() > expiry:
                del self._memory[mapping_id]
                return None
            
            return mapping
    
    def cleanup_expired(self) -> int:
        """Remove expired mappings. Returns count removed."""
        if self._redis:
            return 0  # Redis handles TTL automatically
        
        now = time.time()
        expired = [k for k, (_, exp) in self._memory.items() if now > exp]
        for k in expired:
            del self._memory[k]
        return len(expired)


# =============================================================================
# Unified Helper Functions (used by both Chat UI and Platform)
# =============================================================================

def attach_mapping_to_message(
    message: dict,
    mapping: dict[str, str],
    anonymized_text: str,
) -> dict:
    """Attach anonymization mapping to a gradio message metadata.
    
    Unified helper for Chat UI path.
    
    Args:
        message: Gradio message dict (will be mutated)
        mapping: PII mapping dict
        anonymized_text: The anonymized text sent to LLM
        
    Returns:
        The same message dict (mutated)
    """
    metadata = message.get("metadata") or {}
    metadata["anonymization"] = {
        "mapping": mapping,
        "anonymized_for_llm": anonymized_text,
    }
    message["metadata"] = metadata
    return message


def extract_mapping_from_message(message: dict) -> dict[str, str] | None:
    """Extract anonymization mapping from gradio message metadata.
    
    Unified helper for Chat UI path.
    
    Args:
        message: Gradio message dict
        
    Returns:
        Mapping dict or None if not present
    """
    metadata = message.get("metadata") or {}
    anon_data = metadata.get("anonymization") or {}
    return anon_data.get("mapping")


def attach_mapping_id_to_message(
    message: dict,
    mapping_id: str,
) -> dict:
    """Attach lightweight mapping ID to message (for space efficiency).
    
    Alternative to full mapping attachment - stores only ID.
    
    Args:
        message: Gradio message dict (will be mutated)
        mapping_id: Mapping store ID
        
    Returns:
        The same message dict (mutated)
    """
    metadata = message.get("metadata") or {}
    metadata["anonymization"] = {
        "mapping_id": mapping_id,
    }
    message["metadata"] = metadata
    return message


# =============================================================================
# Pipeline Extensions (add to pipeline.py)
# =============================================================================

# In AnonymizationPipeline, add these methods:

"""
    def anonymize_with_id(
        self, 
        text: str, 
    ) -> tuple[str, str]:
        '''Anonymize text and return (anonymized_text, mapping_id).
        
        Unified API - stores mapping and returns ID.
        
        Args:
            text: Input text to anonymize
            
        Returns:
            Tuple of (anonymized_text, mapping_id)
        '''
        anonymized_text, mapping = self.anonymize(text)
        
        if mapping:
            mapping_id = self.mapping_store.store(mapping)
            return anonymized_text, mapping_id
        
        return anonymized_text, None
    
    def deanonymize_with_id(
        self, 
        text: str, 
        mapping_id: str,
    ) -> DeanonymizationResult:
        '''Deanonymize using mapping ID.
        
        Unified API - retrieves mapping by ID.
        
        Args:
            text: Anonymized text
            mapping_id: Mapping ID from anonymize_with_id
            
        Returns:
            DeanonymizationResult
        '''
        mapping = self.mapping_store.get(mapping_id)
        return self.deanonymize(text, mapping=mapping)
"""
```

#### 4.0.2 Unified Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED ANONYMIZATION FLOW                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: user_message                                               │
│     │                                                              │
│     ▼                                                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ anonymize(text) → (anonymized_text, mapping)               │    │
│  │        │                                                      │    │
│  │        ▼                                                      │    │
│  │   mapping_store.store(mapping) → mapping_id                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│     │                                                              │
│     ▼                                                              │
│  anonymized_text ──► LLM ──► response (may contain placeholders)  │
│     │                                                              │
│     │  ┌─────────────────────────────────────────────────────┐    │
│     │  │ CASE 1: Chat UI (multi-turn)                       │    │
│     │  │   - Attach mapping_id to message metadata          │    │
│     │  │   - Store for subsequent deanonymization            │    │
│     │  └─────────────────────────────────────────────────────┘    │
│     │  ┌─────────────────────────────────────────────────────┐    │
│     │  │ CASE 2: Platform (single turn)                     │    │
│     │  │   - Return mapping directly to caller               │    │
│     │  │   - Caller passes to deanonymize()                  │    │
│     │  └─────────────────────────────────────────────────────┘    │
│     ▼                                                              │
│  deanonymize(response, mapping_id | mapping) ──► final_answer    │
│     │                                                              │
│     ▼                                                              │
│  OUTPUT: deanonymized text                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.1 Integration in `rag_engine/api/app.py`

#### Before Guardian Check (around line 959):

```python
# Add to imports
from rag_engine.anonymization.pipeline import AnonymizationPipeline
from rag_engine.anonymization.mapping_store import (
    attach_mapping_to_message,
    attach_mapping_id_to_message,
)
from rag_engine.config.settings import settings

# Initialize pipeline (lazy, at module level)
_anonymization_pipeline: AnonymizationPipeline | None = None

def _get_anonymization_pipeline() -> AnonymizationPipeline | None:
    """Get or create anonymization pipeline."""
    global _anonymization_pipeline
    if _anonymization_pipeline is None:
        if settings.anonymizer_enabled:
            _anonymization_pipeline = AnonymizationPipeline()
    return _anonymization_pipeline

# In agent_chat_handler, before guardian check:
anonymizer = _get_anonymization_pipeline()
pii_mapping = {}
mapping_id = None

if anonymizer:
    anonymized_message, mapping = anonymizer.anonymize(message)
    pii_mapping = mapping  # Keep for deanonymization
    
    if mapping:
        # Store mapping by ID for multi-turn retrieval
        mapping_id = anonymizer.mapping_store.store(mapping)
        
        # Attach to gradio history message (Chat UI path)
        # This happens when message is added to gradio_history
        attach_mapping_id_to_message(gradio_history[-1], mapping_id)
        
        # Use anonymized message for LLM
        message = anonymized_message
        
    logger.info(f"Anonymized message: {len(pii_mapping)} entities replaced")
```

#### After LLM Response (streaming):

```python
# In stream handling, before yielding to user:
if anonymizer and pii_mapping:
    # Deanonymize final answer using stored mapping
    result = anonymizer.deanonymize(answer, mapping=pii_mapping)
    answer = result.deanonymized_text
    
    if result.remaining_placeholders:
        logger.warning(
            f"Deanonymization: {len(result.remaining_placeholders)} "
            f"placeholders not restored: {result.remaining_placeholders[:5]}"
        )
```

### 4.2 Platform Integration Flow

For Comindware platform (single turn: request → answer):

```python
# In platform API handler (uses same unified API)
async def handle_platform_request(request: dict) -> dict:
    original_text = request.get("message", "")
    
    anonymizer = _get_anonymization_pipeline()
    
    if anonymizer:
        # Step 1: Anonymize - unified API returns (text, mapping)
        anonymized_text, mapping = anonymizer.anonymize(original_text)
        
        # Step 2: Process with LLM (anonymized_text)
        response = await process_with_llm(anonymized_text)
        
        # Step 3: Deanonymize - unified API accepts mapping directly
        result = anonymizer.deanonymize(response, mapping=mapping)
        
        return {"answer": result.deanonymized_text}
    else:
        # Normal processing (anonymization disabled)
        return await process_with_llm(original_text)
```

### 4.3 Multi-turn Chat UI

For regular chat UI with multiple turns, using unified helpers:

```python
# In gradio history structure (unified format):
# {
#     "role": "user",
#     "content": "original message",  # User sees original
#     "metadata": {
#         "log": "...",
#         "anonymization": {
#             "mapping_id": "anon_abc123def456"  # Lightweight reference
#         }
#     }
# }

# For assistant messages (to track which placeholders were used):
# {
#     "role": "assistant", 
#     "content": "deanonymized answer",
#     "metadata": {
#         "anonymization": {
#             "used_placeholders": ["[NAME_0]", "[EMAIL_0]"]
#         }
#     }
# }

# Retrieving mapping for multi-turn context:
def _get_mapping_for_message(message: dict, anonymizer) -> dict | None:
    """Unified helper to get mapping from message metadata."""
    from rag_engine.anonymization.mapping_store import extract_mapping_from_message
    
    # Try direct mapping first
    mapping = extract_mapping_from_message(message)
    if mapping:
        return mapping
    
    # Try mapping_id reference
    metadata = message.get("metadata") or {}
    anon_data = metadata.get("anonymization") or {}
    mapping_id = anon_data.get("mapping_id")
    
    if mapping_id and anonymizer:
        return anonymizer.mapping_store.get(mapping_id)
    
    return None
```

---

## 5. Inference Provider Factory

Support multiple inference providers for model serving:

### 5.1 Factory (`rag_engine/anonymization/serving/factory.py`)

```python
"""Inference provider factory for anonymization models."""

from abc import ABC, abstractmethod
from typing import Any


class InferenceProvider(ABC):
    """Base interface for inference providers."""
    
    @abstractmethod
    def __call__(self, text: str) -> list[dict]:
        """Run inference on text."""
        pass


class LocalProvider(InferenceProvider):
    """Direct torch/transformers inference."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        from transformers import pipeline
        self.pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )
    
    def __call__(self, text: str) -> list[dict]:
        return self.pipeline(text)


class VLLMProvider(InferenceProvider):
    """vLLM HTTP API inference."""
    
    def __init__(self, url: str, model: str, api_key: str = "EMPTY", **kwargs):
        import openai
        self.client = openai.OpenAI(base_url=url, api_key=api_key)
        self.model = model
    
    def __call__(self, text: str) -> list[dict]:
        # Format for NER endpoint (vLLM may need custom endpoint)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"Extract entities: {text}"}],
        )
        # Parse response - depends on model output format
        return self._parse_response(response)
    
    def _parse_response(self, response) -> list[dict]:
        # Implementation depends on model
        raise NotImplementedError("vLLM NER parsing depends on model")


class MOSECProvider(InferenceProvider):
    """MOSEC HTTP API inference."""
    
    def __init__(self, endpoint: str, model: str, **kwargs):
        # Similar to VLLM but with MOSEC-specific handling
        pass


def create_provider(
    provider_type: str,
    model_name: str,
    **kwargs
) -> InferenceProvider:
    """Factory function to create inference provider."""
    providers = {
        "local": LocalProvider,
        "vllm": VLLMProvider,
        "mosec": MOSECProvider,
    }
    
    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return provider_class(model_name=model_name, **kwargs)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests Structure

```
rag_engine/tests/test_anonymization/
├── __init__.py
├── test_pipeline.py           # Main pipeline tests
├── test_stages.py             # Individual stage tests
├── test_reversibility.py     # Round-trip anonymization/deanonymization
├── test_integration.py       # Full flow integration tests
└── fixtures/
    ├── __init__.py
    ├── samples.py            # Test samples from notebook
    └── datasets/
        └── jayguard_sample.json  # Subset of jayguard-ner-benchmark
```

### 6.2 Test Samples

From `Anonymization.ipynb`:

```python
# Russian samples with PII
RUSSIAN_SAMPLES = [
    "Иван Иванов, +7-900-123-45-67, ivan.ivanov@example.com, Москва, ул. Ленина, д. 10, кв. 5. Работает менеджером в ООО 'Рога и Копыта'.",
    "Анна Смирнова, +7-901-234-56-78, anna.smirnova@company.ru, Санкт-Петербург, Невский пр-т, д. 25, оф. 10. Является директором в 'ТехноПрогресс'.",
    # ... more samples
]

# Corporate sensitive samples
CORPORATE_SAMPLES = [
    "Дмитрий Смирнов, директор по продажам в ООО 'Технострой', тел: +7-905-111-22-33, dmitry.smirnov@tekhnostroy.ru, адрес: г. Санкт-Петербург, Невский пр-т, д. 100.",
    "Пароль от системы: SuperSecret123",
    "API ключ для интеграции: sk_live_abc123xyz",
    "Внутренний ресурс: http://intranet.company.local/dashboard",
    # ... more samples
]
```

### 6.3 Key Test Cases

1. **Reversibility Test**: `anonymize(deanonymize(text)) == text`
2. **Overlap Handling**: No double-replacement of overlapping entities
3. **Edge Cases**: Empty text, no PII, all PII, nested PII
4. **Multi-turn**: Mapping persists across conversation turns
5. **LLM Output**: Handle when LLM uses placeholders in response

---

## 7. Edge Cases & Error Handling

### 7.1 Overlapping Entities

**Issue**: Email "ivan.ivanov@example.com" detected as:
- Email: "ivan.ivanov@example.com"
- Also matches NAME pattern "Иван Иванович" (false positive overlap)

**Solution**:
1. Sort entities by confidence (high → low)
2. Process in reverse position order
3. Skip if overlap with already-processed entity

### 7.2 Nested Placeholders

**Issue**: LLM outputs "[NAME_0] wrote to [EMAIL_0]"

**Solution**:
- Sort placeholders by length (reverse) before replacement
- Use word boundaries where possible

### 7.3 Partial Deanonymization

**Issue**: LLM modifies placeholder: "[NAME_0]" → "[NAME_0s]" or "[NAME_0]!"

**Solution**:
1. Exact match first
2. Fuzzy match for common suffixes (only at end of placeholder)
3. Log warning for unrestored placeholders

### 7.4 Model Inference Failures

**Issue**: EU-PII-Safeguard or Russian NER model unavailable

**Solution**:
1. Fall back to previous stage only
2. Log warning, continue with remaining stages
3. Ensure at least regex stage works offline

---

## 8. Implementation Phases

### Phase 1: Core Pipeline (Week 1)
- [ ] Create directory structure
- [ ] Implement data types and base interfaces
- [ ] Implement Stage 1: Regex + Presidio
- [ ] Basic reversibility
- [ ] Integration in `app.py` (before guardian)
- [ ] Unit tests for Stage 1

### Phase 2: ML Models (Week 2)
- [ ] Implement Stage 2: EU-PII-Safeguard
- [ ] Implement Stage 3: Russian NER
- [ ] Inference provider factory
- [ ] vLLM/MOSEC integration
- [ ] Unit tests for all stages

### Phase 3: Polish & Integration (Week 3)
- [ ] Multi-turn session management
- [ ] Platform integration flow
- [ ] LLM wrapper prompt
- [ ] Deanonymization of LLM output
- [ ] Integration tests
- [ ] Performance benchmarks

### Phase 4: Production Hardening (Week 4)
- [ ] Error handling and fallbacks
- [ ] Logging and debugging
- [ ] Configuration validation
- [ ] Documentation
- [ ] Regression test suite

---

## 9. Open Questions

1. ~~**PII Mapping Storage**: Use message metadata (`log` field) or runtime context?~~
   - **DECIDED**: Unified `MappingStore` class with:
     - `store(mapping) -> mapping_id` - returns ID for storage
     - `get(mapping_id) -> mapping` - retrieves by ID
     - Helper functions: `attach_mapping_to_message()`, `extract_mapping_from_message()`
     - Both Chat UI and Platform use same API

2. **Stage Configuration**: All 3 stages enabled by default?
   - Recommendation: Yes, configurable via `.env`

3. **Deanonymization Edge Cases**: How to handle modified placeholders?
   - Recommendation: Log warnings, provide admin dashboard for audit

4. **Performance Overhead**: EU-PII-Safeguard adds ~200-500ms
   - Acceptable for compliance; add to estimated response time

5. **HuggingFace Token**: EU-PII-Safeguard requires terms acceptance
   - Document in setup instructions; consider using HF_TOKEN env var

6. **Latin vs Russian NER**: EU-PII-Safeguard vs Russian-specific NER?
   - EU-PII-Safeguard works for Russian too (see reference notebook)
   - Three stages: (1) deterministic/regex, (2) EU NER (multilingual), (3) Russian NER (Gherman/Natasha)
   - Stage 3 complements Stage 2 for Russian-specific entities

7. **Russian NER Gaps**: What entities does Gherman miss vs rus-anonymizer?
   - rus-anonymizer: Passport, SNILS, INN, car numbers, etc.
   - These are covered by Stage 1 regex
   - Gherman complements with PERSON/LOC/ORG detection

8. **Config Format**: YAML vs .env?
   - Recommendation: `.env` for enable/disable + secrets, `anonymization.yaml` for structured config

9. **Placeholder Format**: Generic `[NAME_0]` vs Semantic "Person A"?
   - **DECIDED**: Use semantic placeholders ("Person A", "Email B") based on DataAnonymiser approach
   - Benefits: More readable for LLM, consistent mapping (same original → same placeholder)
   - Two-phase approach: Phase 1 counts entities, Phase 2 replaces consistently


---

## 10. References

- [rus-anonymizer](https://github.com/JohnConnor123/rus-anonymizer)
- [ru-smb-pd-anonymizer](https://github.com/ranas-mukminov/ru-smb-pd-anonymizer)
- [DataAnonymiser](https://github.com/levitation-opensource/DataAnonymiser) - Semantic placeholders, two-phase replacement
- [Microsoft Presidio](https://microsoft.github.io/presidio/)
- [tabularisai/eu-pii-safeguard](https://huggingface.co/tabularisai/eu-pii-safeguard)
- [Gherman/bert-base-NER-Russian](https://huggingface.co/Gherman/bert-base-NER-Russian)
- [just-ai/jayguard-ner-benchmark](https://huggingface.co/datasets/just-ai/jayguard-ner-benchmark)
- [cmw-mosec](https://github.com/arterm-sedov/cmw-mosec)
- [cmw-vllm](https://github.com/arterm-sedov/cmw-vllm)

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-27
