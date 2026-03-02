# Anonymization Pipeline Implementation Plan

**Version:** 1.2 | **Date:** 2026-03-01 | **Status:** Draft for Review  
**Author:** OpenCode Agent  
**Based on:** User requirements + rus-anonymizer + ru-smb-pd-anonymizer + Microsoft Presidio + tabularisai/eu-pii-safeguard + Gherman/bert-base-NER-Russian + Just AI Jay Guard research

**Changes (v1.2):** Fixed DetectionStage enum order; Removed duplicate OKVED/OKPO; Fixed Phase 1 description (no Presidio); Removed duplicate helpers.py; Added patterns from ru-smb-pd-anonymizer, rus-anonymizer; Added benchmarking section from Jay Guard

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
│  Stage 1: Direct Regex Patterns (No Presidio)               │
│  - Fast (0.1ms/sample), Reliable for structured IDs         │
│  - Russian phones, emails, passports, SNILS, INN, credentials│
│  - Why no Presidio: 200x slower, fewer detections          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Russian NER (Gherman/bert-base-NER-Russian)      │
│  - Moderate (50-70ms/sample on CPU)                         │
│  - Highly accurate for Russian Names, Locations, Orgs        │
│  - spaCy ru_core_news_lg NOT used: not trained for PII     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Multilingual NER (tabularisai/eu-pii-safeguard)  │
│  - Slow (~150-200ms/sample on CPU)                          │
│  - Multilingual fallback for missed entities                  │
│  - 42 entity types (Jobs, Addresses, etc.)                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Post-Processing: Merge & Resolve Overlaps                  │
│  - Merge adjacent same-semantic-type entities                 │
│  - Resolve overlaps: Longest Match First                     │
│  - Normalize to unified semantic names                        │
└─────────────────────────────────────────────────────────────┘
    ↓
[Guardian Check - Content Moderation - existing]
    ↓
[LLM Processing - with anonymized content]
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Deanonymization                                           │
│  - Restore original PII from mapping                        │
│  - Handle LLM output containing placeholders               │
└─────────────────────────────────────────────────────────────┘
    ↓
User sees original text (transparent experience)

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
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py              # Base stage interface
│   │   ├── stage1_regex.py       # Direct regex patterns (no Presidio)
│   │   ├── stage2_gherman.py     # Gherman Russian NER
│   │   └── stage3_eu_pii.py      # EU-PII-Safeguard
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── factory.py           # Inference provider factory
│   │   ├── vllm_client.py       # vLLM HTTP client
│   │   ├── mosec_client.py     # MOSEC HTTP client
│   │   └── local.py            # Direct torch inference
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
    ├── test_anonymization_stages.py  # Manual testing script (tested: works!)
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

# Stage 1: Direct Regex Patterns (No Presidio)
# =============================================================================
# Stage 1 uses direct Python regex patterns - NO Presidio framework.
# Why no Presidio:
# - Testing showed Presidio + spaCy ru_core_news_lg: 6 detections, 19.7ms
# - Direct regex: 17 detections, 0.1ms (200x faster, 3x more accurate)
# - spaCy Russian models not trained for PII detection
# - Direct regex gives full control without framework overhead
#
# Patterns cover (borrowed from rus-anonymizer, ru-smb-pd-anonymizer):
# - Russian phones (+7, 8, multiple formats)
# - Russian passports, SNILS, INN, OGRN, KPP, OKPO, OKVED
# - Car numbers, VIN, OMS policy, Driver license
# - Corporate credentials (password, login, API keys, internal URLs)
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
    
    # Medical/Professional (borrowed from rus-anonymizer)
    - OMS_POLICY
    - DRIVER_LICENSE
    
    # Sensitive/Corporate
    - PASSWORD
    - LOGIN
    - API_KEY
    - INTERNAL_URL
    
    # Optional - uncomment if needed
    # - DATE_TIME
    # - BANK_CARD
    # - BIRTH_CERT
    # - MILITARY_ID
    # - OKATO
    # - BLOOD_TYPE
  
# Direct Regex Patterns
# =============================================================================
# These patterns run directly (no Presidio framework).
# Each pattern is a Python regex string.
# Additional patterns borrowed from ru-smb-pd-anonymizer, rus-anonymizer
# =============================================================================
  patterns:
    # Russian phone patterns (6 formats from rus-anonymizer)
    phone_1: '\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'  # +7 (XXX) XXX-XX-XX
    phone_2: '\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'      # +7 XXX XXX-XX-XX
    phone_3: '8\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'    # 8 (XXX) XXX-XX-XX
    phone_4: '8\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}'        # 8 XXX XXX-XX-XX
    phone_5: '\+7\d{10}'                                        # +7XXXXXXXXXX
    phone_6: '8\d{10}'                                          # 8XXXXXXXXXX
    # Additional phone formats (borrowed)
    phone_7: '\+7[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}'  # +7 XXX XXX XX XX with flexible separators
    phone_8: '7\d{10}'                                           # 7XXXXXXXXXX (no leading +)
    
    # Email
    email: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Russian documents (borrowed from ru-smb-pd-anonymizer)
    passport: '\b\d{2}\s+\d{2}\s+\d{6}\b|\b\d{4}\s+\d{6}\b'
    snils: '\b\d{3}[-\s]*\d{3}[-\s]*\d{3}[-\s]*\d{2}\b'
    inn: '\b\d{10}\b|\b\d{12}\b'
    ogrn: '\b\d{13}\b|\b\d{15}\b'
    kpp: '\b\d{9}\b'
    okpo: '\b\d{8}\b|\b\d{10}\b'
    okved: '\b\d{2}\.\d{2}(\.\d{1,2})?\b'
    
    # Medical/Professional (borrowed from rus-anonymizer)
    oms_policy: '\b\d{9}\b'                                      # OMS policy - 9 digits
    driver_license: '\b\d{2}\s*\d{2}\s*\d{6}\b'                  # Driver license
    birth_cert: '\bМС-\d{6}\b|\b\d{2}-\d{2}-\d{6}\b'            # Birth certificate
    
    # Financial
    bic: '\b0[4-5]\d{7}\b'
    bank_account: '\b40[5-8]\d{17}\b|\b301\d{17}\b'
    credit_card: '\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    
    # Vehicle
    car_number: '\b[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}\b'
    vin: '\b[A-HJ-NPR-Z0-9]{17}\b'
    
    # Corporate sensitive
    password: '(пароль|password|passwd|pwd)[\s:=_]+(\S+)'
    login: '(логин|login|username)[\s:=_]+(\S+)'
    api_key: '(api[_-]?key|apikey)[\s:=_]+(\S+)'
    internal_url: '(https?)://(intranet|internal|local|localhost|192\.168|10\.|172\.(1[6-9]|2|3[01])|127\.0\.0\.1)[^\s,]*'
    url: 'https?://[^\s/$.?#].[^\s,]*'
    ip_address: '\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    
    # Note: AGE detection handled by NER models (Stage 2: Gherman, Stage 3: EU-PII)
    # Regex age patterns are too rigid/unreliable for Russian

# =============================================================================
# US/ENGLISH PATTERNS (Stage 1, optional - for English text)
# =============================================================================
# These patterns supplement Russian patterns for US/English PII detection.
# They can be enabled/disabled independently via config.
# =============================================================================
stage1_regex_us:
  enabled: false  # Enable for English text processing
  
  # US-specific patterns
  patterns:
    # US Phone (multiple formats)
    us_phone_1: '\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'  # (555) 123-4567, 555-123-4567, +1...
    us_phone_2: '\+?\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}'  # International
    
    # US Social Security Number
    us_ssn: '\b\d{3}-\d{2}-\d{4}\b'
    
    # US Individual Taxpayer Identification Number
    us_itin: '\b9\d{2}-\d{7}\b'
    
    # US Employer Identification Number
    us_ein: '\b\d{2}-\d{7}\b'
    
    # US ZIP Code
    us_zip: '\b\d{5}(-\d{4})?\b'
    
    # US Driver License (generic - varies by state)
    us_driver_license: '\b[A-Z]{1,2}\d{5,8}\b'
    
    # US DEA Number
    us_dea: '\b[A-Z]{2}\d{7}\b'
    
    # US Passport
    us_passport: '\bA\d{8}\b'
    
    # US NPI (National Provider Identifier)
    us_npi: '\b[1-2]\d{9}\b'
    
    # US ABA Routing Number
    us_aba_routing: '\b\d{9}\b'
    
    # US Street Address (common suffixes)
    us_street_address: '\b\d+\s+[A-Za-z0-9\s,.]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Way|Lane|Ln|Pkwy|Circle|Cir)\b'
    
    # US State (abbreviations)
    us_state: '\b(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b'

# Stage 2: Gherman Russian NER
# =============================================================================
# Gherman/bert-base-NER-Russian provides accurate detection of:
# - Russian names (PERSON, FIRST_NAME, LAST_NAME)
# - Russian locations (CITY, REGION, COUNTRY)
# - Russian organizations
#
# Why not spaCy ru_core_news_lg:
# - spaCy models not trained for PII detection
# - Gherman specifically fine-tuned for Russian NER
# - Better precision/recall for Russian names and locations
# =============================================================================
stage2_gherman:
  enabled: true
  
  # Model configuration
  model:
    # HuggingFace model ID
    name: "Gherman/bert-base-NER-Russian"
    
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
      model: "Gherman/bert-base-NER-Russian"
      api_key: "EMPTY"
  
  # Entity mapping (Gherman output → unified semantic name)
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

# Stage 3: EU-PII-Safeguard
# =============================================================================
# Multilingual NER as final fallback:
# - 42 entity types across 26 European languages
# - Catches entities missed by Stage 1 (regex) and Stage 2 (Gherman)
# - Works well for non-Russian text mixed in Russian documents
# =============================================================================
stage3_eu_pii:
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

# Stage 2: Gherman Russian NER
ANONYMIZER_GHERMAN_PROVIDER=local    # local | vllm | mosec

# Stage 3: EU-PII-Safeguard (requires HuggingFace token acceptance)
ANONYMIZER_EU_PII_PROVIDER=local    # local | vllm | mosec

# Inference endpoints (if using vllm/mosec)
GHERMAN_VLLM_URL=http://localhost:8000/v1
GHERMAN_VLLM_MODEL=Gherman/bert-base-NER-Russian
EU_PII_VLLM_URL=http://localhost:8000/v1
EU_PII_VLLM_MODEL=tabularisai/eu-pii-safeguard

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
anonymizer_gherman_provider: str
anonymizer_eu_pii_provider: str
gherman_vllm_url: str
gherman_vllm_model: str
eu_pii_vllm_url: str
eu_pii_vllm_model: str
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
    STAGE2_GHERMAN = 2
    STAGE3_EU_PII = 3


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
    
    # Additional Russian identifiers (from research - rus-anonymizer, ru-smb-pd-anonymizer)
    OMS_POLICY = "OMS_POLICY"     # Medical insurance policy (9 digits)
    DRIVER_LICENSE = "DRIVER_LICENSE"  # Driver's license
    BIRTH_CERT = "BIRTH_CERT"     # Birth certificate (МС-ХХХХХХ format)
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
    
    # US/English identifiers
    US_SSN = "US_SSN"
    US_ITIN = "US_ITIN"
    US_EIN = "US_EIN"
    US_ZIP = "US_ZIP"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    US_DEA = "US_DEA"
    US_PASSPORT = "US_PASSPORT"
    US_NPI = "US_NPI"
    US_ABA_ROUTING = "US_ABA_ROUTING"
    US_STREET_ADDRESS = "US_STREET_ADDRESS"
    US_STATE = "US_STATE"
    
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

### 3.3 Stage 1: Direct Regex Patterns (`rag_engine/anonymization/stages/stage1_regex.py`)

```python
"""Stage 1: Direct regex patterns (no Presidio framework).

This stage uses direct Python regex patterns for maximum speed and control.
Why no Presidio:
- Testing showed Presidio + spaCy ru_core_news_lg: 6 detections, 19.7ms
- Direct regex: 17 detections, 0.1ms (200x faster, 3x more accurate)
- spaCy Russian models not trained for PII detection
- Direct regex gives full control without framework overhead
"""

import re
import logging
from typing import Any

from rag_engine.anonymization.stages.base import AnonymizationStage
from rag_engine.anonymization.types import (
    DetectionStage,
    DetectedEntity,
    EntityType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Regex Patterns (Direct Python - No Presidio)
# =============================================================================

class RegexPatterns:
    """Direct regex patterns for Russian PII detection."""
    
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
    PASSWORD_PATTERN = re.compile(r'(пароль|password|passwd|pwd)[\s:=_]+(\S+)', re.IGNORECASE)
    LOGIN_PATTERN = re.compile(r'(логин|login|username)[\s:=_]+(\S+)', re.IGNORECASE)
    API_KEY_PATTERN = re.compile(r'(api[_-]?key|apikey)[\s:=_]+(\S+)', re.IGNORECASE)
    INTERNAL_URL_PATTERN = re.compile(r'(https?)://(intranet|internal|local|localhost|192\.168|10\.|172\.(1[6-9]|2|3[01])|127\.0\.0\.1)[^\s,]*', re.IGNORECASE)
    URL_PATTERN = re.compile(r'https?://[^\s/$.?#].[^\s,]*', re.IGNORECASE)
    IP_ADDRESS_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    
    # Note: AGE detection handled by NER models (Stage 2: Gherman, Stage 3: EU-PII)
    
    @classmethod
    def detect_all(cls, text: str) -> list[DetectedEntity]:
        """Detect all PII using regex patterns with overlap handling."""
        entities = []
        
        def add_if_no_overlap(pattern, entity_type, context_keywords=None):
            for match in pattern.finditer(text):
                overlapping = any(
                    not (match.end() <= e.start or match.start() >= e.end)
                    for e in entities
                )
                if overlapping:
                    continue
                if context_keywords:
                    ctx_start = max(0, match.start() - 20)
                    context = text[ctx_start:match.end()].lower()
                    if not any(kw in context for kw in context_keywords):
                        continue
                entities.append(DetectedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,  # Regex is high confidence
                    stage=DetectionStage.STAGE1_REGEX,
                ))
        
        # Priority order (high confidence first)
        add_if_no_overlap(cls.BANK_ACCOUNT_PATTERN, EntityType.BANK_ACCOUNT)
        add_if_no_overlap(cls.VIN_PATTERN, EntityType.VIN)
        add_if_no_overlap(cls.EMAIL_PATTERN, EntityType.EMAIL)
        add_if_no_overlap(cls.PASSWORD_PATTERN, EntityType.PASSWORD)
        add_if_no_overlap(cls.LOGIN_PATTERN, EntityType.LOGIN)
        add_if_no_overlap(cls.API_KEY_PATTERN, EntityType.API_KEY)
        
        for pattern in cls.PHONE_PATTERNS:
            add_if_no_overlap(pattern, EntityType.PHONE)
        add_if_no_overlap(cls.CREDIT_CARD_PATTERN, EntityType.BANK_CARD)
        
        add_if_no_overlap(cls.SNILS_PATTERN, EntityType.SNILS)
        add_if_no_overlap(cls.PASSPORT_PATTERN, EntityType.PASSPORT)
        add_if_no_overlap(cls.CAR_NUMBER_PATTERN, EntityType.CAR_NUMBER)
        
        # IDs that need context
        add_if_no_overlap(cls.INN_PATTERN, EntityType.INN, ['инн', 'inn', 'налог'])
        add_if_no_overlap(cls.OGRN_PATTERN, EntityType.OGRN, ['огрн', 'ogrn'])
        add_if_no_overlap(cls.KPP_PATTERN, EntityType.KPP, ['кпп', 'kpp'])
        add_if_no_overlap(cls.BIC_PATTERN, EntityType.BIC, ['бик', 'bic'])
        
        add_if_no_overlap(cls.URL_PATTERN, EntityType.URL)
        add_if_no_overlap(cls.IP_ADDRESS_PATTERN, EntityType.IP_ADDRESS)
        
        # Note: AGE is handled by NER models, not regex
        
        return sorted(entities, key=lambda x: x.start)


class RegexAnonymizationStage(AnonymizationStage):
    """Stage 1: Direct regex patterns (no Presidio)."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.patterns = RegexPatterns()
    
    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect PII using direct regex patterns."""
        return self.patterns.detect_all(text)
```

### 3.4 Stage 2: Gherman Russian NER (`rag_engine/anonymization/stages/stage2_gherman.py`)

```python
"""Stage 2: Gherman Russian NER model.

Gherman/bert-base-NER-Russian provides accurate detection of:
- Russian names (PERSON, FIRST_NAME, LAST_NAME)
- Russian locations (CITY, REGION, COUNTRY)
- Russian organizations

Why not spaCy ru_core_news_lg:
- spaCy models not trained for PII detection
- Gherman specifically fine-tuned for Russian NER
"""

from typing import Any

from rag_engine.anonymization.stages.base import AnonymizationStage
from rag_engine.anonymization.types import (
    DetectionStage,
    DetectedEntity,
    EntityType,
)

# Entity mapping from Gherman output to unified types
GHERMAN_ENTITY_MAP = {
    "FIRST_NAME": EntityType.NAME,
    "MIDDLE_NAME": EntityType.NAME,
    "LAST_NAME": EntityType.NAME,
    "PER": EntityType.NAME,
    "PERSON": EntityType.NAME,
    "CITY": EntityType.ADDRESS,
    "LOC": EntityType.ADDRESS,
    "LOCATION": EntityType.ADDRESS,
    "REGION": EntityType.ADDRESS,
    "COUNTRY": EntityType.ADDRESS,
    "ORGANIZATION": EntityType.COMPANY,
    "ORG": EntityType.COMPANY,
}


class GhermanAnonymizationStage(AnonymizationStage):
    """Stage 2: Gherman Russian NER."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.ner_pipeline = None
        self.device = config.get("model", {}).get("device", "auto")
        self.confidence_threshold = config.get("model", {}).get("confidence_threshold", 0.5)
    
    def _load_model(self):
        """Lazy load the model."""
        if self.ner_pipeline is None:
            from transformers import pipeline
            model_name = self.config.get("model", {}).get("name", "Gherman/bert-base-NER-Russian")
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=self.device,
            )
    
    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect Russian PII using Gherman NER."""
        self._load_model()
        
        results = self.ner_pipeline(text)
        entities = []
        
        for r in results:
            entity_group = r.get("entity_group", "")
            if entity_group == "O" or not entity_group:
                continue
            
            if r.get("score", 0) < self.confidence_threshold:
                continue
            
            entity_type = GHERMAN_ENTITY_MAP.get(entity_group)
            if entity_type is None:
                continue
            
            entities.append(DetectedEntity(
                text=r.get("word", ""),
                entity_type=entity_type,
                start=r.get("start", 0),
                end=r.get("end", 0),
                confidence=r.get("score", 0),
                stage=DetectionStage.STAGE2_GHERMAN,
            ))
        
        return entities
```

### 3.5 Stage 3: EU-PII-Safeguard (`rag_engine/anonymization/stages/stage3_eu_pii.py`)

```python
"""Stage 3: EU-PII-Safeguard multilingual NER.

Multilingual NER as final fallback:
- 42 entity types across 26 European languages
- Catches entities missed by Stage 1 (regex) and Stage 2 (Gherman)
- Works well for non-Russian text mixed in Russian documents
"""

from typing import Any

from rag_engine.anonymization.stages.base import AnonymizationStage
from rag_engine.anonymization.types import (
    DetectionStage,
    DetectedEntity,
    EntityType,
)

# Entity mapping from EU-PII output to unified types
EU_PII_ENTITY_MAP = {
    "PERSON": EntityType.NAME,
    "EMAIL_ADDRESS": EntityType.EMAIL,
    "PHONE_NUMBER": EntityType.PHONE,
    "LOCATION": EntityType.ADDRESS,
    "ORGANIZATION": EntityType.COMPANY,
}


class EuPiiAnonymizationStage(AnonymizationStage):
    """Stage 3: EU-PII-Safeguard multilingual NER."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.ner_pipeline = None
        self.device = config.get("model", {}).get("device", "auto")
        self.confidence_threshold = config.get("model", {}).get("confidence_threshold", 0.5)
    
    def _load_model(self):
        """Lazy load the model."""
        if self.ner_pipeline is None:
            from transformers import pipeline
            model_name = self.config.get("model", {}).get("name", "tabularisai/eu-pii-safeguard")
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=self.device,
            )
    
    def detect(self, text: str) -> list[DetectedEntity]:
        """Detect multilingual PII using EU-PII-Safeguard."""
        self._load_model()
        
        results = self.ner_pipeline(text)
        entities = []
        
        for r in results:
            entity_group = r.get("entity_group", "")
            if entity_group == "O" or not entity_group:
                continue
            
            if r.get("score", 0) < self.confidence_threshold:
                continue
            
            entity_type = EU_PII_ENTITY_MAP.get(entity_group)
            if entity_type is None:
                continue
            
            entities.append(DetectedEntity(
                text=r.get("word", ""),
                entity_type=entity_type,
                start=r.get("start", 0),
                end=r.get("end", 0),
                confidence=r.get("score", 0),
                stage=DetectionStage.STAGE3_EU_PII,
            ))
        
        return entities
```

### 3.4 Pipeline Orchestrator (`rag_engine/anonymization/pipeline.py`)

```python
"""Main anonymization pipeline orchestrator.

Runs all 3 stages in parallel on original text, then merges/deduplicates results.
"""

import logging
from typing import Any

from rag_engine.anonymization.config import AnonymizationConfig
from rag_engine.anonymization.mapping_store import MappingStore
from rag_engine.anonymization.stages.stage1_regex import RegexAnonymizationStage
from rag_engine.anonymization.stages.stage2_gherman import GhermanAnonymizationStage
from rag_engine.anonymization.stages.stage3_eu_pii import EuPiiAnonymizationStage
from rag_engine.anonymization.types import (
    AnonymizationResult,
    DeanonymizationResult,
    DetectedEntity,
)

logger = logging.getLogger(__name__)


class AnonymizationPipeline:
    """Cascaded reversible anonymization pipeline.
    
    Runs multiple stages in parallel on original text, then merges/deduplicates.
    Each stage sees the full original text to maximize detection.
    """
    
    def __init__(self, config: AnonymizationConfig | None = None):
        """Initialize pipeline with configuration."""
        self.config = config or AnonymizationConfig()
        
        # Initialize stages (all run on original text in parallel)
        self.stages = []
        
        if self.config.stage1_enabled:
            self.stages.append(RegexAnonymizationStage(self.config.stage1_config))
        
        if self.config.stage2_enabled:
            self.stages.append(GhermanAnonymizationStage(self.config.stage2_config))
        
        if self.config.stage3_enabled:
            self.stages.append(EuPiiAnonymizationStage(self.config.stage3_config))
        
        # Initialize mapping store
        self.mapping_store = MappingStore(ttl=self.config.session_mapping_ttl)
        
        logger.info(f"AnonymizationPipeline initialized with {len(self.stages)} stages")
    
    def anonymize(
        self, 
        text: str, 
    ) -> tuple[str, dict]:
        """Anonymize text and return (anonymized_text, mapping).
        
        All stages run on original text in parallel, then results are merged/deduplicated.
        
        Args:
            text: Input text to anonymize
            
        Returns:
            Tuple of (anonymized_text, pii_mapping)
        """
        if not text or not text.strip():
            return text, {}
        
        # Run all stages in parallel on original text
        all_entities: list[DetectedEntity] = []
        
        for stage in self.stages:
            if stage.is_enabled():
                try:
                    entities = stage.detect(text)
                    all_entities.extend(entities)
                except Exception as e:
                    logger.warning(f"Stage {stage.__class__.__name__} failed: {e}")
        
        # Merge/deduplicate: all stages see original text
        entities = self._resolve_overlaps(all_entities, text)
        
        if not entities:
            logger.debug("No PII entities detected")
            return text, {}
        
        # Generate placeholders and apply replacements
        mapping: dict[str, str] = {}
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        
        anonymized = text
        for entity in sorted_entities:
            placeholder = f"[{entity.entity_type.value}_{len(mapping)}]"
            mapping[placeholder] = entity.text
            anonymized = anonymized[:entity.start] + placeholder + anonymized[entity.end:]
        
        logger.info(f"Anonymized {len(entities)} entities")
        
        return anonymized, mapping
    
    def deanonymize(
        self, 
        text: str, 
        mapping: dict[str, str] | None = None,
        mapping_id: str | None = None,
    ) -> DeanonymizationResult:
        """Restore original PII from anonymized text."""
        # Get mapping from store if mapping_id provided
        if mapping_id and not mapping:
            mapping = self.mapping_store.get(mapping_id)
        
        if not mapping:
            return DeanonymizationResult(
                anonymized_text=text,
                deanonymized_text=text,
                restored_count=0,
            )
        
        # Sort by length (reverse) to avoid partial replacements
        sorted_placeholders = sorted(mapping.keys(), key=len, reverse=True)
        
        deanonymized = text
        restored_count = 0
        
        for placeholder in sorted_placeholders:
            if placeholder in deanonymized:
                deanonymized = deanonymized.replace(placeholder, mapping[placeholder])
                restored_count += 1
        
        return DeanonymizationResult(
            anonymized_text=text,
            deanonymized_text=deanonymized,
            restored_count=restored_count,
        )
    
    def _resolve_overlaps(
        self, 
        entities: list[DetectedEntity],
        text: str,
    ) -> list[DetectedEntity]:
        """Merge adjacent and deduplicate overlapping entities.
        
        Algorithm:
        1. Sort by length (longest first)
        2. Use covered indices to track what's replaced
        3. Merge adjacent same-type entities (fragmentation control)
        """
        if not entities:
            return []
        
        # Sort by length (longest first), then by start
        sorted_entities = sorted(
            entities, 
            key=lambda x: (x.end - x.start, -x.start), 
            reverse=True
        )
        
        text_len = len(text)
        covered = [False] * text_len
        final = []
        
        for e in sorted_entities:
            start, end = e.start, e.end
            
            # Check if overlaps with already covered
            is_overlapping = any(covered[idx] for idx in range(start, min(end, text_len)))
            
            if not is_overlapping:
                final.append(e)
                for idx in range(start, min(end, text_len)):
                    covered[idx] = True
        
        # Merge adjacent same-type entities
        final.sort(key=lambda x: x.start)
        merged = []
        
        for e in final:
            if not merged:
                merged.append(e)
                continue
            
            prev = merged[-1]
            # Adjacent (within 2 chars) AND same semantic type
            is_adjacent = prev.end >= e.start - 2 and prev.entity_type == e.entity_type
            
            if is_adjacent:
                # Merge: extend previous
                prev.text = prev.text + ' ' + e.text
                prev.end = e.end
                # Keep higher confidence
                if e.confidence > prev.confidence:
                    prev.confidence = e.confidence
            else:
                merged.append(e)
        
        return merged
```

### 3.5 LLM Wrapper Prompt (`rag_engine/anonymization/prompts.py`)

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
├── test_benchmark.py         # Performance benchmarking (borrowed from Jay Guard)
└── fixtures/
    ├── __init__.py
    ├── samples.py            # Test samples from notebook
    └── datasets/
        └── jayguard_sample.json  # Subset of jayguard-ner-benchmark
```

### 6.2 Benchmarking (borrowed from Jay Guard methodology)

```python
"""Performance benchmarking - borrowed from Jay Guard methodology.

Reference: https://habr.com/ru/companies/just_ai/articles/946392/
Dataset: https://huggingface.co/datasets/just-ai/jayguard-ner-benchmark
"""

import time
import json
from dataclasses import dataclass
from typing import Callable

@dataclass
class BenchmarkResult:
    """Results from benchmarking a detection stage."""
    stage_name: str
    total_samples: int
    total_time_seconds: float
    avg_latency_ms: float
    detections_per_second: float
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None


def benchmark_stage(
    stage_name: str,
    detect_fn: Callable[[str], list],
    samples: list[str],
    ground_truth: list[dict] | None = None,
) -> BenchmarkResult:
    """Benchmark a detection stage.
    
    Args:
        stage_name: Name of the stage being benchmarked
        detect_fn: Detection function to benchmark
        samples: List of text samples
        ground_truth: Optional list of ground truth annotations for F1 calculation
    
    Returns:
        BenchmarkResult with latency and accuracy metrics
    """
    total_time = 0
    total_detections = 0
    
    for sample in samples:
        start = time.perf_counter()
        entities = detect_fn(sample)
        elapsed = time.perf_counter() - start
        
        total_time += elapsed
        total_detections += len(entities)
    
    avg_latency_ms = (total_time / len(samples)) * 1000
    dps = len(samples) / total_time if total_time > 0 else 0
    
    result = BenchmarkResult(
        stage_name=stage_name,
        total_samples=len(samples),
        total_time_seconds=total_time,
        avg_latency_ms=avg_latency_ms,
        detections_per_second=dps,
    )
    
    # Calculate F1 if ground truth provided
    if ground_truth:
        # Compare detections vs ground truth
        # ... (F1 calculation logic)
        pass
    
    return result


# Expected benchmarks (based on Jay Guard article):
# - Regex: ~0.1ms/sample, very high precision
# - Gherman Russian NER: ~50-70ms/sample on CPU
# - EU-PII-Safeguard: ~150-200ms/sample on CPU
#
# Jay Guard benchmarks for reference (from article):
# | Model        | PERSON F1 | PERSON Precision | PERSON Recall |
# |--------------|-----------|------------------|----------------|
# | Natasha      | 0.64      | 0.63             | 0.68           |
# | spaCy ru    | 0.73      | 0.73             | 0.76           |
# | DeepPavlov   | 0.58      | 0.59             | 0.59           |
# | flair       | 0.86      | 0.86             | 0.87           |
# | GLiNER      | 0.45      | 0.44             | 0.53           |
```

### 6.3 Test Samples

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
- [ ] Implement Stage 1: Direct Regex Patterns (no Presidio)
- [ ] Basic reversibility
- [ ] Integration in `app.py` (before guardian)
- [ ] Unit tests for Stage 1

### Phase 2: ML Models (Week 2)
- [ ] Implement Stage 2: Gherman Russian NER
- [ ] Implement Stage 3: EU-PII-Safeguard
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
   - Three stages: (1) deterministic/regex, (2) Russian NER (Gherman), (3) EU NER (multilingual)
   - Stage 3 complements Stage 2 for non-Russian entities

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
   - **VERIFIED**: Tested in `test_anonymization_stages.py` - working approach


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
